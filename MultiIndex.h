//SPDX-License-Identifier: MIT-0
#ifndef MI_MULTI_INDEX_H
#define MI_MULTI_INDEX_H
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>
#include <memory>
#include <vector>
#include <stdexcept>
#include <atomic>
#include <optional>
#include <concepts>
#include <variant>

namespace ns_mi {
   /*
   ========================================================================================================================
   mi::T_MultiIndex — Design notes and architecture
   ========================================================================================================================
   
   Goals
   -----
   - Zero-overhead when only a single (primary) index is used; secondary indices add cost only when present.
   - Stable "handle" abstraction (T_Handle) providing non-owning pointers to elements with configurable behavior.
   - Support for non-movable payloads without penalizing movable types (e.g., std::atomic support).
   - Allocator-aware with careful propagation semantics matching STL conventions.
   - Optional lazy deletion via tombstones for high-churn scenarios.
   
   Policy dimensions (orthogonal traits controlled by policy selection)
   --------------------------------------------------------------------
   Each policy controls four compile-time boolean traits:
   
     c_invalidates:              Does primary storage relocate elements? (e.g., vector vs map)
     c_stores_handle:            Do secondaries store T_Handle directly? (vs key or ordinal)
     c_needs_translation_array:  Do we maintain a central ordinal->node* translation array?
     c_uses_tombstones:          Does erase() mark nodes "dead" vs physical removal?
   
   Built-in policies and their trade-offs
   ---------------------------------------
   - S_NoInv:                  For node-stable primaries (std::map). Fastest, simplest.
                               c_invalidates=false, c_stores_handle=true
   
   - S_UpdatePointerPolicy:    For relocating primaries. Patches all secondary handles on move.
                               c_invalidates=true, c_stores_handle=true
                               Cost: O(#affected secondaries) per relocation
   
   - S_TranslationArrayPolicy: For frequent relocations. Secondaries store ordinals.
                               c_invalidates=true, c_needs_translation_array=true
                               Cost: O(1) per relocation, extra indirection on access
   
   - S_KeyLookupPolicy:        Secondaries store primary keys. Requires unique primary.
                               c_invalidates=false, c_stores_handle=false
                               Cost: Primary lookup on every secondary access
   
   Handle/Iterator stability guarantees
   ------------------------------------
   - T_Handle: Non-owning pointer to primary node
     • tombstones=false: erase() invalidates handle immediately
     • tombstones=true: erase() marks dead; handle remains readable but not writable
   - Primary iterators: Live-skipping when tombstones enabled
   - Secondary iterators: May translate stored values to handles on-the-fly
   
   Exception safety model
   ----------------------
   - emplace/modify/replace: Strong guarantee via "drop-rebuild" pattern
   - Secondary rollback on failure restores original state
   - Assumption: DropSecs() must not throw for consistency
   
   Threading (conditionally thread-safe subset)
   --------------------------------------------
   Safe concurrent operations when ALL conditions met:
   - Operations: Only emplace() + find()/contains() (no erase/modify/iteration)
   - Containers: All indices use concurrent, node-stable maps (e.g., PPL concurrent_unordered_map)
   - Policy restrictions:
     • OK: S_NoInv, S_KeyLookupPolicy
     • OK: S_UpdatePointerPolicy ONLY if primary never relocates at runtime
     • NOT OK: S_TranslationArrayPolicy (unsynchronized vector)
   - If tombstones enabled: Use I_PerThreadErr=true for atomic live counter
   
   Visibility: emplace() publishes to primary first, then secondaries. Brief window
   where primary contains element but secondary doesn't. No linearizability across indices.
   
   Allocator handling
   ------------------
   - Primary map's allocator propagates per standard traits
   - Secondary maps may have incompatible allocator_type; MakeMap() ignores mismatches
   - Translation array rebinds primary allocator to void* when present
   - Copy/move assignment respects propagate_on_container_* traits
   
   Known limitations and sharp edges
   ----------------------------------
   - WrapToPair uses offsetof(std::pair, second) - assumes standard-layout
   - Modify/replace rebuilds ALL secondaries (not minimal-delta)
   - With tombstones: memory grows until compact() called (O(N) operation)
   - S_UpdatePointerPolicy: O(bucket_size) scan per relocation in worst case
   */
   
   //###############################################################################################################################################
   // Concepts (C++20)
   // Design note: These concepts enable compile-time branching without SFINAE complexity.
   // HasAllocator detects allocator-aware containers to handle propagation correctly.
   //###############################################################################################################################################
   
   template<class P_T>
   concept HasAllocator = requires(const P_T& t) {
      typename P_T::allocator_type;
      { t.get_allocator() };
   };

   //###############################################################################################################################################
   // Rationale: Visual Studio workarounds - these should be inline variable templates but compiler bugs
   // force us to use inline functions. They determine allocator propagation behavior at compile time.
   //###############################################################################################################################################
   template<class P_Map> //Have to define this and the next one separate because of Visual Studio bugs...
   inline constexpr bool ReassignAllocatorOnCopy() {
      if constexpr (HasAllocator<P_Map>)
         return std::allocator_traits<typename P_Map::allocator_type>::propagate_on_container_copy_assignment::value;
      else
         return false;
   }
   template<class P_Map>
   inline constexpr bool ReassignAllocatorOnMove() {
      if constexpr (HasAllocator<P_Map>)
         return std::allocator_traits<typename P_Map::allocator_type>::propagate_on_container_move_assignment::value;
      else
         return false;
   }
   template<class P_Map>
   inline constexpr bool FastMoveDisallowed() {
      if constexpr (HasAllocator<P_Map>)
         return !std::allocator_traits<typename P_Map::allocator_type>::propagate_on_container_move_assignment::value;
      else
         return false;
   }
   
   // emits a compile-time *warning* when instantiated with <false>
   // Design note: This warns when an allocator is passed but ignored by non-allocator-aware maps.
   // We deliberately don't force rebinding - maps have fixed allocator_type and forcing a mismatch would be brittle.
   template<bool I_OK> struct S_MiAllocatorCheck { };

   template<>
   struct [[deprecated("mi::multi_index: the allocator you passed is ignored by one or more index maps that do not support allocators")]]
      S_MiAllocatorCheck<false> { };

   template<class P_T>
   concept O_EqualityComparable = requires(const P_T& a, const P_T& b) { { a == b } -> std::convertible_to<bool>; };

   //###############################################################################################################################################
   // Helpers for supporting non-movable types
   // Design rationale: Many high-performance containers assume movability. We explicitly support copy-only types
   // (e.g., std::atomic) without penalizing movable types. This avoids forcing users to wrap non-movables.
   //###############################################################################################################################################
   template<class P_T>
   concept O_Movable = std::is_move_constructible_v<P_T>;

   template<class P_T>
   concept O_CopyOnly = std::is_copy_constructible_v<P_T> && !O_Movable<P_T>;

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Helper to move if movable, otherwise copy
   // Rationale: Transparently degrades to copy for non-movables, allowing uniform code paths
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   template<class T>
   decltype(auto) conditional_move(T& v) {
      if constexpr (O_Movable<T>)
         return std::move(v);
      else
         return v;
   }

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Helper to forward arguments correctly for copy-only types
   // Takes the template parameter T for what type to forward as, and the actual value
   // Rationale: Prevents binding rvalue refs to deleted move ctors for copy-only types
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   template<class T, class U>
   decltype(auto) safe_forward(U&& v) {
      using raw_t = std::remove_reference_t<T>;
      if constexpr (!std::is_lvalue_reference_v<T> && O_CopyOnly<raw_t>)
         return static_cast<const raw_t&>(v);
      else
         return std::forward<T>(v);
   }

   //###############################################################################################################################################
   // Rationale: MakeMap deliberately accepts potentially incompatible allocators.
   // If types mismatch, we ignore it for that map (most containers have fixed allocator_type).
   // Policies still receive the original allocator for their auxiliary structures.
   //###############################################################################################################################################
   template<class P_Map, class P_Alloc, class... P_Args>
   decltype(auto) MakeMap(const P_Alloc& a, P_Args&&... args) {
      if constexpr (!std::is_same_v<P_Alloc, std::monostate>) {
         return P_Map(std::forward<P_Args>(args)...);
      }
      else if constexpr (HasAllocator<P_Map>) {
         // Maps have fixed allocator types. We can only use the allocator if it matches.
         using map_alloc_t = typename P_Map::allocator_type;

         // Only use the allocator if it's compatible with what the map expects
         if constexpr (std::is_same_v<std::decay_t<P_Alloc>, map_alloc_t>) {
            // Exact match - rare but possible
            return P_Map(a, std::forward<P_Args>(args)...);
         }
         else {
            // Different allocator type - expected and normal
            // The allocator will be used by the policy instead
            // Suppress the warning since this is intentional
            // S_MiAllocatorCheck<false>{};  // Optional: uncomment if you want the warning
            return P_Map(std::forward<P_Args>(args)...);
         }
      }
      else
         return P_Map(std::forward<P_Args>(args)...);
   }

   //###############################################################################################################################################
   // misc meta helpers (simplified)
   // Design note: These compute index properties at compile time using fold expressions,
   // avoiding runtime overhead for index management.
   //###############################################################################################################################################
   template<class... P_S> inline constexpr std::size_t primary_count = (static_cast<std::size_t>(P_S::is_storage) + ... + 0);

   // Simplified with fold expression
   template<class... P_S>
   inline constexpr std::size_t stor_pos_v = []<std::size_t... I_Is>(std::index_sequence<I_Is...>) {
      constexpr bool is_storage[] = {std::tuple_element_t<I_Is, std::tuple<P_S...>>::is_storage...};
         for (std::size_t i = 0; i < sizeof...(P_S); ++i)
            if (is_storage[i]) 
               return i;
      return size_t(0); // fallback
   }(std::make_index_sequence<sizeof...(P_S)>{});

   template<class P_Tag, class... P_S>
   inline constexpr std::size_t tag_pos_v = []<std::size_t... I_Is>(std::index_sequence<I_Is...>) {
      std::size_t pos = static_cast<std::size_t>(-1);
      ( (void)(
         std::is_same_v<typename std::tuple_element_t<I_Is, std::tuple<P_S...>>::t_tag, P_Tag>
            ? (pos = I_Is, 0)
            : 0
         ), ... );
      return pos;
   }(std::index_sequence_for<P_S...>{});

   template<class C>
   concept O_UniqueAssociativeContainer = requires (std::remove_cvref_t<C>& c, typename std::remove_cvref_t<C>::value_type v) {
      { c.insert(std::move(v)).second } -> std::convertible_to<bool>;
      // .second is only present when the return type is std::pair<iterator,bool>
   };

   template<class C>
   inline constexpr bool is_unique_v = O_UniqueAssociativeContainer<C>;

   //###############################################################################################################################################
   // Base that conditionally exposes the standard allocator typedefs
   // Design rationale: Expose allocator traits only when primary has them, maintaining STL compatibility
   // while avoiding spurious typedefs for non-allocator-aware primaries.
   //###############################################################################################################################################
   template<class P_Primary, bool I_HasAlloc = HasAllocator<P_Primary>>
   struct S_AllocFacadeBase { };

   template<class P_Primary>
   struct S_AllocFacadeBase<P_Primary, true> {
      using t_allocator_type = typename P_Primary::allocator_type;
      using t_propagate_on_container_copy_assignment =
         typename std::allocator_traits<t_allocator_type>::propagate_on_container_copy_assignment;
      using t_propagate_on_container_move_assignment =
         typename std::allocator_traits<t_allocator_type>::propagate_on_container_move_assignment;
      using t_propagate_on_container_swap =
         typename std::allocator_traits<t_allocator_type>::propagate_on_container_swap;
   };

   //###############################################################################################################################################
   // Base classes for translation array index storage and empty optimization
   // Rationale: C_TranslBase holds the translation array index when needed by policy.
   // Using inheritance allows Empty Base Optimization when not needed.
   //###############################################################################################################################################
   
   class C_TranslBase {
      template<class P_Derived, bool I_Tombs>
      friend struct S_TranslationArrayPolicyFull;

      std::size_t trans_idx{0};

   public:
      C_TranslBase() = default;
      C_TranslBase(C_TranslBase&& rhs) noexcept : trans_idx(rhs.trans_idx) {
      }
      C_TranslBase& operator=(C_TranslBase&& rhs) noexcept {
         if (this != &rhs)
            trans_idx = rhs.trans_idx;
         return *this;
      }
   };
   class C_EmptyBase {
   };

   //###############################################################################################################################################
   // Live counter base class - conditionally contains m_live based on c_tombstones
   // Design rationale: Tombstones allow lazy deletion - elements are marked dead but not removed.
   // This enables: 1) Node reuse for same-key reinserts, 2) Stable iteration under erase.
   // The live counter ensures size() remains O(1) despite dead nodes in storage.
   // I_PerThreadErr makes the counter atomic for read-only concurrent access (size() queries).
   //###############################################################################################################################################
   template<bool I_Tombstones, bool I_PerThreadErr>
   class T_LiveCounterBase {
   protected:
      T_LiveCounterBase() noexcept = default;
      T_LiveCounterBase(const T_LiveCounterBase&) noexcept = default;
      T_LiveCounterBase(T_LiveCounterBase&&) noexcept = default;
      T_LiveCounterBase& operator=(const T_LiveCounterBase&) noexcept = default;
      T_LiveCounterBase& operator=(T_LiveCounterBase&&) noexcept = default;
      void IncrementLive() noexcept { }
      void DecrementLive() noexcept { }
      void SetLive(std::size_t) noexcept { }
      std::size_t GetLive() const noexcept { return 0; }
      void SwapLive(T_LiveCounterBase&) noexcept { }
   };

   template<bool I_PerThreadErr>
   class T_LiveCounterBase<true, I_PerThreadErr> {
   private:
      // Atomic counter for concurrent size() reads when I_PerThreadErr=true
      std::conditional_t<I_PerThreadErr, std::atomic<std::size_t>, std::size_t> m_live{0};

   protected:
      T_LiveCounterBase() noexcept = default;
      
      T_LiveCounterBase(const T_LiveCounterBase& rhs) noexcept {
         if constexpr (I_PerThreadErr)
            m_live.store(rhs.m_live.load());
         else
            m_live = rhs.m_live;
      }
      T_LiveCounterBase& operator=(const T_LiveCounterBase& rhs) noexcept {
         if constexpr (I_PerThreadErr) {
            if (this != &rhs)
               m_live.store(rhs.m_live.load());
         }
         else
            m_live = rhs.m_live;
         return *this;
      }
      void IncrementLive() noexcept { ++m_live; }
      void DecrementLive() noexcept { --m_live; }

      void SetLive(std::size_t n) noexcept { 
         if constexpr (I_PerThreadErr) 
            m_live.store(n);
         else 
            m_live = n;
      }

      std::size_t GetLive() const noexcept { 
         if constexpr (I_PerThreadErr) 
            return m_live.load();
         else 
            return m_live;
      }

      void SwapLive(T_LiveCounterBase& other) noexcept {
         if constexpr (I_PerThreadErr) {
            std::size_t cur_live = m_live.load();
            m_live.store(other.m_live.load());
            other.m_live.store(cur_live);
         } else
            std::swap(m_live, other.m_live);
      }
   };

   //###############################################################################################################################################
   // T_PayloadWrap: Encapsulates user payload with policy-required metadata
   // Design: Three specializations optimize memory based on policy needs:
   // 1. Non-invalidating: Just the payload (zero overhead)
   // 2. Invalidating without tombstones: Add owner pointer for relocation callbacks
   // 3. Invalidating with tombstones: Add both owner pointer and dead flag
   // The owner pointer enables synchronous notification on element relocation.
   //###############################################################################################################################################
   template<class P_P, bool I_Tombs, bool I_Invalidates, bool I_NeedsTransl, class P_PolicyType>
   class T_PayloadWrap;

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Non-invalidating version
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   template<class P_P, class P_PolicyType>
   class T_PayloadWrap<P_P, /*tombs*/ false, /*invalidates*/false, /*needs tranlation*/ false, P_PolicyType> {
      template<class P_Pair> friend class T_Handle;
      template<class P_Key, class P_Payload, bool I_PerThreadErr, template<class> class P_Policy, class... P_Specs>
      friend class T_MultiIndex;

      P_P m_value;
   public:
      template<class... P_Args>
      explicit T_PayloadWrap(P_Args&&... args) : m_value(safe_forward<P_Args>(args)...) { }

      T_PayloadWrap(const T_PayloadWrap&) = default;
      T_PayloadWrap& operator=(const T_PayloadWrap&) = default;

      T_PayloadWrap(T_PayloadWrap&& rhs) : m_value(conditional_move(rhs.m_value)) { }
      T_PayloadWrap& operator=(T_PayloadWrap&& rhs) {
         if (this != &rhs)
            m_value = conditional_move(rhs.m_value);
         return *this;
      }

      auto const& Payload() const noexcept { return m_value; }
      const P_P* operator->() const noexcept { return std::addressof(m_value); }
      const P_P& operator*()  const noexcept { return m_value; }
      operator const P_P&()  const noexcept { return m_value; }
   };

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // No tombstone version (also invalidates)
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   template<class P_P, bool I_NeedsTransl, class P_PolicyType>
   class T_PayloadWrap<P_P, /*tombs*/false, /*invalidates*/true, I_NeedsTransl, P_PolicyType> : public std::conditional_t<I_NeedsTransl, C_TranslBase, C_EmptyBase> {
      template<class P_Pair> friend class T_Handle;
      template<class P_Key, class P_Payload, bool I_PerThreadErr, template<class> class P_Policy, class... P_Specs>
      friend class T_MultiIndex;

      using t_base = std::conditional_t<I_NeedsTransl, C_TranslBase, C_EmptyBase>;

      P_P m_value;
      P_PolicyType* m_p_owner{nullptr};  // still needed when c_invalidates==true

   public:
      template<class... P_Args>
      explicit T_PayloadWrap(P_Args&&... args) : m_value(safe_forward<P_Args>(args)...) { }

      T_PayloadWrap(const T_PayloadWrap&) = delete;
      T_PayloadWrap& operator=(const T_PayloadWrap&) = delete;

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Critical: Move operations trigger OnRelocate callback to maintain secondary coherence
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      T_PayloadWrap(T_PayloadWrap&& rhs) : t_base(std::move(rhs)), m_value(conditional_move(rhs.m_value)), m_p_owner(rhs.m_p_owner) {
         if (m_p_owner && &rhs != this)
            m_p_owner->OnRelocate(&rhs, this);
      }

      T_PayloadWrap& operator=(T_PayloadWrap&& rhs) {
         if (this != &rhs) {
            this->t_base::operator=(std::move(rhs));
            m_value = conditional_move(rhs.m_value);
            m_p_owner = rhs.m_p_owner;
            if (m_p_owner && &rhs != this)
               m_p_owner->OnRelocate(&rhs, this);
         }
         return *this;
      }
      auto const& Payload() const noexcept { return m_value; }
      const P_P* operator->() const noexcept { return std::addressof(m_value); }
      const P_P& operator*()  const noexcept { return m_value; }
      operator const P_P&()  const noexcept { return m_value; }
   };

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // With tombstones version (always invalidates)
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   template<class P_P, bool I_NeedsTransl, class P_PolicyType>
   class T_PayloadWrap<P_P, /*tombs*/true, /*invalidates*/true, I_NeedsTransl, P_PolicyType> : public std::conditional_t<I_NeedsTransl, C_TranslBase, C_EmptyBase> {
      template<class P_Pair> friend class T_Handle;
      template<class P_Key, class P_Payload, bool I_PerThreadErr, template<class> class P_Policy, class... P_Specs>
      friend class T_MultiIndex;

      using t_base = std::conditional_t<I_NeedsTransl, C_TranslBase, C_EmptyBase>;

      P_P m_value;
      bool dead{false};
      P_PolicyType* m_p_owner{nullptr};

   public:
      template<class... P_Args>
      explicit T_PayloadWrap(P_Args&&... args) : m_value(safe_forward<P_Args>(args)...) { }

      T_PayloadWrap(const T_PayloadWrap&) = delete;
      T_PayloadWrap& operator=(const T_PayloadWrap&) = delete;

      T_PayloadWrap(T_PayloadWrap&& rhs) : t_base(std::move(rhs)), m_value(conditional_move(rhs.m_value)), dead(rhs.dead), m_p_owner(rhs.m_p_owner) {
         if (m_p_owner && &rhs != this)
            m_p_owner->OnRelocate(&rhs, this);
      }

      T_PayloadWrap& operator=(T_PayloadWrap&& rhs) {
         if (this != &rhs) {
            this->t_base::operator=(std::move(rhs));
            m_value = conditional_move(rhs.m_value);
            dead = rhs.dead;
            m_p_owner = rhs.m_p_owner;
            if (m_p_owner && &rhs != this)
               m_p_owner->OnRelocate(&rhs, this);
         }
         return *this;
      }
      bool IsDead() const noexcept { return dead; }
      auto const& Payload() const noexcept { return m_value; }
      const P_P* operator->() const noexcept { return std::addressof(m_value); }
      const P_P& operator*()  const noexcept { return m_value; }
      operator const P_P&()  const noexcept { return m_value; }
   };

   //###############################################################################################################################################
   // Relocation policies
   //
   // POLICY ARCHITECTURE OVERVIEW:
   // Policies use CRTP to inject behavior into T_MultiIndex. Each policy defines:
   // 1. How secondary indices reference primary nodes (handle/key/ordinal)
   // 2. How to maintain coherence when primary storage relocates elements
   // 3. Whether to use tombstones for lazy deletion
   // 
   // The four boolean traits create 16 possible combinations, though only a few make sense.
   // Built-in policies cover the common, performant patterns.
   //
   //###############################################################################################################################################
   template<bool I_Invalidates, bool I_NeedsTransl, bool I_StoresHandle, bool I_UsesTombstones>
   struct T_PolicyBase {
      static constexpr bool c_invalidates = I_Invalidates;
      static constexpr bool c_needs_translation_array = I_NeedsTransl;
      static constexpr bool c_stores_handle = I_StoresHandle;
      static constexpr bool c_uses_tombstones = I_UsesTombstones;

      // Default no-op implementations
      template<class P_WrapType>
      void OnRelocate(P_WrapType*, P_WrapType*) { }

      template<class P_HandleType, class P_PairType>
      void OnEmplaceSuccess(P_HandleType, P_PairType*) { }

      void OnEmplaceFail() { }

      template<class P_Self>
      void PolicySwap(P_Self&) noexcept { }

      void PolicyClear() { }
   };

   //###############################################################################################################################################
   // S_NoInv: Optimal for node-stable primaries (std::map, std::unordered_map)
   // Secondaries store T_Handle directly. No relocation handling needed.
   // Thread-safety: Safe for concurrent emplace+find with concurrent containers.
   //###############################################################################################################################################
   template<class P_Derived>
   struct S_NoInv : T_PolicyBase<false, false, true, false> {
      S_NoInv() = default;
      template<class P_Alloc> explicit S_NoInv(const P_Alloc&) noexcept {}
      template<class P_Alloc> void PolicyResetAllocator(const P_Alloc&) noexcept {}

      template<class P_HandleType>
      static P_HandleType ToHandle(const P_HandleType& h) noexcept { return h; }

      template<class P_HandleType, class P_Iterator>
      bool MatchSecondary(P_Iterator it, P_HandleType h) const { return it->second == h; }

      template<class P_HandleType>
      auto GetSecondaryValue(P_HandleType h) const { return h; }
   };

   //###############################################################################################################################################
   // S_UpdatePointerPolicyFull: For relocating primaries (vector, deque)
   // Secondaries store T_Handle; OnRelocate patches all affected handles.
   // Cost: O(#secondaries ? bucket_size) per relocation in worst case.
   // Thread-safety: Safe for concurrent ops ONLY if primary never actually relocates.
   //###############################################################################################################################################
   template<class P_Derived, bool I_Tombs = false>
   struct S_UpdatePointerPolicyFull : T_PolicyBase<true, false, true, I_Tombs> {
      S_UpdatePointerPolicyFull() = default;
      template<class P_Alloc> explicit S_UpdatePointerPolicyFull(const P_Alloc&) noexcept {}
      template<class P_Alloc> void PolicyResetAllocator(const P_Alloc&) noexcept {}

      template<class P_HandleType>
      static P_HandleType ToHandle(const P_HandleType& h) noexcept { return h; }

      template<class P_WrapType>
      void OnRelocate(P_WrapType* p_old_wrap, P_WrapType* p_new_wrap) {
         auto& self = static_cast<P_Derived&>(*this);
         auto* p_old_pair = self.WrapToPair(p_old_wrap);
         auto* p_new_pair = self.WrapToPair(p_new_wrap);

         using t_handle = typename P_Derived::t_handle;

         // Build handles before/after relocation
         t_handle old_h{p_old_pair};
         t_handle new_h{p_new_pair};

         // Walk every secondary map and patch the stored handle in-place
         self.ForEachSecondary([&]<std::size_t I_I>() -> bool {
            using spec  = typename P_Derived::template t_spec<I_I>;
            // Skip the primary map
            if constexpr (!spec::is_storage) {
               auto& m   = self.template idx<I_I>();
               auto skey = spec::Project(new_h.Key(), std::as_const(new_h).Payload());
               auto [f, l] = m.equal_range(skey);
               for (auto it = f; it != l; ++it)
                  if (it->second == old_h)
                     it->second = new_h;
            }
            return true;   // continue fold-expression
         });
      }

      template<class P_HandleType, class P_Iterator>
      bool MatchSecondary(P_Iterator it, P_HandleType h) const {
         return it->second == h;
      }

      template<class P_HandleType>
      auto GetSecondaryValue(P_HandleType h) const { return h; }
   };

   template<class P_Derived>
   using S_UpdatePointerPolicy = S_UpdatePointerPolicyFull<P_Derived, false>;

   template<class P_Derived>
   using S_UpdatePointerPolicyTombs = S_UpdatePointerPolicyFull<P_Derived, true>;

   //###############################################################################################################################################
   // Helper for rebinding allocator to void* for translation array
   // Forward declaration to avoid premature instantiation
   //###############################################################################################################################################
   template<class P_Derived, bool HasAllocator>
   struct S_TranslAllocHelper {
      using type = std::allocator<void*>;
   };

   // Specialization for when allocator exists
   template<class P_Derived>
   struct S_TranslAllocHelper<P_Derived, true> {
      using type = typename std::allocator_traits<typename P_Derived::t_allocator_type>::template rebind_alloc<void*>;
   };

   //###############################################################################################################################################
   // S_TranslationArrayPolicyFull: Optimal for frequent relocations
   // Secondaries store ordinals; central translation array maps ordinal?node*.
   // Cost: O(1) per relocation, extra indirection on access.
   // WARNING: NOT thread-safe for concurrent ops (unsynchronized vector).
   //###############################################################################################################################################
   template<class P_Derived, bool I_Tombs = false>
   struct S_TranslationArrayPolicyFull : T_PolicyBase<true, true, false, I_Tombs> {
      // Does the derived expose an allocator via S_AllocFacadeBase?
      static constexpr bool c_has_alloc = requires { typename P_Derived::t_allocator_type; };

      // Rebind the primary allocator (if present) to void* for the transl array.
      // Design: Ensures translation array uses same memory resource as primary.
      using t_transl_alloc = typename S_TranslAllocHelper<P_Derived, c_has_alloc>::type;
      mutable std::vector<void*, t_transl_alloc> transl;

      S_TranslationArrayPolicyFull() = default;

      // Accept allocator from T_MultiIndex and use it for the translation array.
      template<class P_Alloc>
      explicit S_TranslationArrayPolicyFull(const P_Alloc& a) : transl([&]() {
         if constexpr (c_has_alloc) return t_transl_alloc(a);
         else return t_transl_alloc();
      }()) {
      }

      template<class P_WrapType>
      void OnRelocate(P_WrapType* p_old_wrap, P_WrapType* p_new_wrap) {
         auto& self = static_cast<P_Derived&>(*this);
         auto* p_old_pair = self.WrapToPair(p_old_wrap);
         auto* p_new_pair = self.WrapToPair(p_new_wrap);
         transl[p_old_pair->second.trans_idx] = p_new_pair;
      }

      template<class P_HandleType>
      P_HandleType ToHandle(std::size_t idx) const {
         return P_HandleType{transl[idx]};
      }

      template<class P_HandleType, class P_Iterator>
      bool MatchSecondary(P_Iterator it, P_HandleType h) const {
         return transl[it->second] == std::as_const(h).Raw();
      }

      template<class P_HandleType>
      auto GetSecondaryValue(P_HandleType h) const {
         return std::as_const(h).Raw()->second.trans_idx;
      }

      template<class P_HandleType, class P_PairType>
      void OnEmplaceSuccess(P_HandleType, P_PairType* p_p) {
         transl.push_back(p_p);
         static_cast<C_TranslBase&>(p_p->second).trans_idx = transl.size() - 1;
      }

      void OnEmplaceFail() {
         if (!transl.empty())
            transl.pop_back();
      }

      void PolicySwap(S_TranslationArrayPolicyFull& other) noexcept {
         using std::swap;
         swap(transl, other.transl);
      }

      void PolicyClear() {
         transl.clear();
      }

      // Rebind the policy's allocator after a clear(), used by T_MultiIndex on copy-assign
      // when propagate_on_container_copy_assignment == true and allocators differ.
      template<class P_Alloc>
      void PolicyResetAllocator(const P_Alloc& a) {
         if constexpr (c_has_alloc) {
            t_transl_alloc desired(a);
            // If not always-equal, only rebind when actually different.
            if constexpr (!std::allocator_traits<t_transl_alloc>::is_always_equal::value) {
               if (transl.get_allocator() == desired)
                  return;
            }
            std::vector<void*, t_transl_alloc> tmp(desired);
            transl.swap(tmp); // transl is empty at this point (T_MultiIndex calls clear() first)
         }
      }
   };

   template<class P_Derived>
   using S_TranslationArrayPolicy = S_TranslationArrayPolicyFull<P_Derived, false>;
   template<class P_Derived>
   using S_TranslationArrayPolicyTombs = S_TranslationArrayPolicyFull<P_Derived, true>;

   //###############################################################################################################################################
   // S_KeyLookupPolicy: Secondaries store primary keys only
   // Handle materialized via primary.find(key). Requires unique primary.
   // Cost: Primary lookup on every secondary access.
   // Thread-safety: Safe for concurrent emplace+find with concurrent containers.
   //###############################################################################################################################################
   template<class P_Derived>
   struct S_KeyLookupPolicy : T_PolicyBase</*invalidates*/false, /*needs_translation_array*/false, /*stores_handle*/false, /*uses_tombstones*/false> {
      S_KeyLookupPolicy() = default;
      template<class P_Alloc> explicit S_KeyLookupPolicy(const P_Alloc&) noexcept {}
      template<class P_Alloc> void PolicyResetAllocator(const P_Alloc&) noexcept {}

      // Materialize a handle from the stored primary key (used by T_HandleIter in translate=true views).
      template<class P_HandleType, class P_KeyType>
      P_HandleType ToHandle(const P_KeyType& k) const {
         // Primary must be unique so a primary key identifies at most one node.
         static_assert(P_Derived::c_primary_is_unique, "S_KeyLookupPolicy requires a unique primary index.");

         const auto& self  = static_cast<const P_Derived&>(*this);
         const auto& prim  = self.primary();
         auto it = prim.find(k); // unique primary => at most one
         if (it == prim.end())
            return P_HandleType{};
         // T_Handle expects a non-const pair*, mirror T_PrimaryHandleIter's const_cast.
         return P_HandleType{const_cast<typename P_HandleType::t_pair*>(&*it) };
      }

      // Match a secondary entry to a handle when removing/rolling back secondaries.
      template<class P_HandleType, class P_Iterator>
      bool MatchSecondary(P_Iterator it, P_HandleType h) const { return it->second == h.Key(); }

      // What to store into secondary maps for a given handle: the primary key.
      template<class P_HandleType>
      auto GetSecondaryValue(P_HandleType h) const { return h.Key(); }
   };

   //###############################################################################################################################################
   // T_Handle: Non-owning, trivially-copyable pointer to primary node
   // Design: Provides const view to users, mutable access to T_MultiIndex.
   // With tombstones, remains valid after erase for inspection (not modification).
   // Dereferencing outside container APIs is UB.
   //###############################################################################################################################################
   template<class P_Pair>
   class T_Handle {
      template<class P_Key, class P_Payload, bool I_PerThreadErr, template<class> class P_Policy, class... P_Specs>
      friend class T_MultiIndex;
      template<class P_Pair2>
      friend class T_Handle;

   public:
      using t_pair = P_Pair;

   private:
      using t_key_cref     = std::add_const_t<typename P_Pair::first_type>&;
      using t_payload_cref = std::add_const_t<decltype(std::declval<typename P_Pair::second_type>().m_value)>&;
      struct S_Proxy {
         t_key_cref     first;
         t_payload_cref second;
         const S_Proxy* operator->() const noexcept { return this; }
      };

      P_Pair* m_p{nullptr};

   public:
      T_Handle() = default;
      explicit T_Handle(P_Pair* p_p) : m_p(p_p) { }
      explicit T_Handle(void* p_p) : m_p(static_cast<P_Pair*>(p_p)) { }

      template<class P_OtherPair>
      T_Handle(const T_Handle<P_OtherPair>& other)  requires (std::is_convertible_v<P_OtherPair*, P_Pair*> && !std::is_same_v<P_OtherPair, P_Pair>) : m_p(other.m_p) {
      }

      t_key_cref Key() const noexcept { return m_p->first; }
       t_payload_cref Payload() const noexcept { return m_p->second.m_value; }
      const auto& Flag() const noexcept requires requires { m_p->second.dead; } { return m_p->second.dead; }
      const P_Pair* Raw() const noexcept { return m_p; }
      explicit operator bool() const noexcept { return m_p; }

      constexpr S_Proxy operator*() const noexcept { return { m_p->first, m_p->second.m_value }; }
      constexpr S_Proxy operator->() const noexcept {
         return **this; // the prvalue returned by operator*(); lifetime is OK
      }
      friend bool operator==(T_Handle, T_Handle) = default;

   private:
      P_Pair* Raw() noexcept { return m_p; }
      bool& Flag() noexcept requires requires { m_p->second.dead; } { return m_p->second.dead; }
      auto& Payload() noexcept { return m_p->second.m_value; }
   };

   //###############################################################################################################################################
   // T_Index<> - compile-time spec
   // Design: Projection flexibility supports three forms:
   // 1. Member pointer: returns p.*member
   // 2. Unary callable: returns proj(p)
   // 3. Binary callable: returns proj(k, p)
   // This allows secondary indices on any computable property.
   //###############################################################################################################################################
   template<template<class...> class P_Map, auto I_Proj, class P_Tag = void, bool I_Storage = false>
   struct T_Index {
      static constexpr bool is_storage = I_Storage;
      using t_tag = P_Tag;

      template<class P_K, class P_V> using t_map_type = P_Map<P_K, P_V>;

      template<class P_K, class P_P>
      static decltype(auto) Project(const P_K& k, const P_P& p) {
         if constexpr (!I_Storage) {
            if constexpr (std::is_member_object_pointer_v<decltype(I_Proj)>)
               return p.*I_Proj;
            else if constexpr (std::is_invocable_v<decltype(I_Proj), const P_P&>)
               return I_Proj(p);
            else
               return I_Proj(k, p);
         }
         else
            return k;
      }

      template<class P_K, class P_P>
      using t_key_type = std::remove_cvref_t<decltype(Project(
         std::declval<const P_K&>(),
         std::declval<const P_P&>()))>;
   };

   template<template<class...> class P_Map, class P_Tag = void>
   using t_primary_index = T_Index<P_Map, 0, P_Tag, true>;

   template<template<class...> class P_Map, auto I_Proj, class P_Tag = void>
   using t_secondary_index = T_Index<P_Map, I_Proj, P_Tag, false>;

   //###############################################################################################################################################
   // Iterator helpers
   //###############################################################################################################################################
   template<class P_It>
   P_It SkipDead(P_It it, P_It end) {
      while (it != end && it->second.IsDead())
         ++it;
      return it;
   }

   //###############################################################################################################################################
   // Iterator that is **only** instantiated when translation is needed
   // Design: Translates stored values (ordinals/keys) to T_Handle on-the-fly.
   // Used when secondaries don't store handles directly.
   //###############################################################################################################################################
   template<class P_RawIt, class P_Policy, class P_Handle>
   class T_HandleIter {
      using t_raw_pair    = typename P_RawIt::value_type;
      using t_key   = typename t_raw_pair::first_type;

   public:
      using iterator_category = std::forward_iterator_tag;
      using value_type        = std::pair<const t_key, P_Handle>;
      using difference_type   = typename P_RawIt::difference_type;
      using reference         = std::pair<const t_key&, P_Handle>;
      using pointer           = const reference*;

   private:
      P_RawIt             m_it;
      const P_Policy*     m_pol;
      mutable std::optional<reference> m_cache;

   public:
      T_HandleIter() = default;
      T_HandleIter(P_RawIt it, const P_Policy* p) : m_it(it), m_pol(p) { }

      reference operator*()  const { Refresh(); return *m_cache; }
      pointer   operator->() const { Refresh(); return &*m_cache; }

      T_HandleIter& operator++()    { ++m_it; return *this; }
      T_HandleIter  operator++(int) { auto t = *this; ++*this; return t; }
      P_RawIt Raw() const noexcept { return m_it; }

      friend bool operator==(const T_HandleIter& a, const T_HandleIter& b) { return a.m_it == b.m_it; }

   private:
      void Refresh() const { m_cache.emplace(m_it->first, m_pol->template ToHandle<P_Handle>(m_it->second)); }
   };

   //###############################################################################################################################################
   // T_LiveIter - skips "dead" entries when c_tombstones==true
   // Design: Wraps raw iterator to transparently skip tombstoned entries.
   // Ensures iteration only visits live elements without manual filtering.
   //###############################################################################################################################################
   template<class P_It>
   class T_LiveIter {
      P_It m_it, m_end;

      template<class P_It2>
      friend class T_LiveIter;

   public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = typename P_It::value_type;
      using difference_type = typename P_It::difference_type;
      using reference = typename P_It::reference;
      using pointer = typename P_It::pointer;

      T_LiveIter(P_It b, P_It e) noexcept : m_it(SkipDead(b, e)), m_end(e) {
      }
      template<class P_OtherIt>
      T_LiveIter(const T_LiveIter<P_OtherIt>& other)  requires (std::is_convertible_v<P_OtherIt, P_It> && !std::is_same_v<P_OtherIt, P_It>) :
         m_it(other.m_it), m_end(other.m_end) {
      }

      reference operator*() const { return *m_it; }
      pointer operator->() const { return m_it.operator->(); }

      T_LiveIter& operator++() { ++m_it; m_it = SkipDead(m_it, m_end); return *this; }
      T_LiveIter operator++(int) { auto t = *this; ++*this; return t; }

      P_It Raw() const noexcept { return m_it; }
      friend bool operator==(const T_LiveIter& a, const T_LiveIter& b) { return a.m_it == b.m_it; }
   };

   //###############################################################################################################################################
   // T_PrimaryHandleIter - Iterator that materializes a T_Handle for primary map
   // Design: Primary storage doesn't store handles, just pairs. This iterator
   // materializes handles on-demand for uniform interface across all views.
   //###############################################################################################################################################
   template<class P_RawIt, class P_Handle>
   class T_PrimaryHandleIter {
      using t_key   = typename P_RawIt::value_type::first_type;
   public:
      using iterator_category = std::forward_iterator_tag;
      using value_type        = std::pair<const t_key, P_Handle>;
      using difference_type   = typename P_RawIt::difference_type;
      using reference         = std::pair<const t_key&, P_Handle>;
      using pointer           = const reference*;

   private:
      P_RawIt                m_it;
      mutable std::optional<reference> m_cache;

   public:
      T_PrimaryHandleIter() = default;
      explicit T_PrimaryHandleIter(P_RawIt it) : m_it(it) {}

      reference operator*()  const { Refresh(); return *m_cache; }
      pointer   operator->() const { Refresh(); return &*m_cache; }

      T_PrimaryHandleIter& operator++()    { ++m_it; return *this; }
      T_PrimaryHandleIter  operator++(int) { auto t = *this; ++*this; return t; }
      P_RawIt Raw() const noexcept { return m_it; }

      friend bool operator==(const T_PrimaryHandleIter& a, const T_PrimaryHandleIter& b) { return a.m_it == b.m_it; }

   private:
      void Refresh() const { m_cache.emplace(m_it->first, P_Handle{ const_cast<typename P_Handle::t_pair*>(&*m_it) }); }
   };

   //###############################################################################################################################################
   // Unified view - with single pointer to T_MultiIndex and index position
   // Design: T_IndexView provides uniform STL-like interface regardless of:
   // - Whether viewing primary or secondary
   // - Whether tombstones need skipping
   // - Whether translation from stored value to handle is needed
   // Template specialization selects optimal iterator combination.
   //###############################################################################################################################################
   template<class P_MultiIndex, std::size_t I_Index, bool I_SkipDead, bool I_Translate>
   class T_IndexView;

   // Forward declaration for friend declaration
   template<class P_Key, class P_Payload, bool I_PerThreadErr, template<class> class P_Policy, class... P_Specs>
   class T_MultiIndex;

   //###############################################################################################################################################
   // Base implementation for primary or secondary already storing handles (fast)
   //###############################################################################################################################################
   template<class P_MultiIndex, std::size_t I_Index, bool I_SkipDead>
   class T_IndexView<P_MultiIndex, I_Index, I_SkipDead, /*translate*/false> {
      template<class P_Key, class P_Payload, bool I_PerThreadErr, template<class> class P_Policy, class... P_Specs>
      friend class T_MultiIndex;

      using t_mi = std::remove_const_t<P_MultiIndex>;
      using t_map = std::decay_t<decltype(std::declval<P_MultiIndex*>()->template GetMap<I_Index>())>;
      using t_handle = typename t_mi::t_handle;
      using t_raw_it = typename t_map::const_iterator;
      using t_tag = typename t_mi::template t_spec<I_Index>::t_tag;

      static constexpr bool c_is_primary = (I_Index == t_mi::c_prim_idx);
      static constexpr bool c_needs_handle_materialization = c_is_primary && !std::is_same_v<typename t_map::mapped_type, t_handle>;

      P_MultiIndex* m_mi;

   public:
      using key_type = typename t_map::key_type;
      using mapped_type = std::conditional_t<c_needs_handle_materialization, t_handle, typename t_map::mapped_type>;
      using size_type = std::size_t;

      // Conditional iterator selection based on whether we need handle materialization
      using iterator_base = std::conditional_t<I_SkipDead, T_LiveIter<t_raw_it>, t_raw_it>;
      using const_iterator = std::conditional_t<c_needs_handle_materialization, T_PrimaryHandleIter<iterator_base, t_handle>, iterator_base>;
      using iterator = const_iterator;

      explicit T_IndexView(P_MultiIndex* mi) : m_mi(mi) {}

      const_iterator begin() const { return ToIterator(GetMap().begin()); }
      const_iterator end() const { return ToIterator(GetMap().end()); }

      const_iterator find(const key_type& key) const {
         if constexpr (!c_is_primary)
            return ToIterator(GetMap().find(key));
         else
            return const_iterator(m_mi->find(key));
      }

      auto equal_range(const key_type& key) const {
         auto [f, l] = GetMap().equal_range(key);
         return std::make_pair(ToIterator(f), ToIterator(l));
      }
      bool contains(const key_type& key) const { return find(key) != end(); }
      bool empty() const { return size() == 0; }

      size_type size() const {
         if constexpr (!c_is_primary)
            return GetMap().size(); // non-primary indices cannot have dead entries
         else
            return m_mi->size(); // Route to main class size() to leverage live counter
      }

      size_type count(const key_type& key) const {
         if constexpr (c_is_primary)
            return m_mi->count(key);
         else
            return GetMap().count(key);
      }

      size_type erase(const key_type& key) requires (!std::is_const_v<P_MultiIndex>) { return m_mi->template erase<t_tag>(key); }
      const_iterator erase(const_iterator it) requires (!std::is_const_v<P_MultiIndex>) { return m_mi->template erase<t_tag>(it); }

      template<class P_Fn>
      bool modify(const_iterator it, P_Fn&& fn) requires (!std::is_const_v<P_MultiIndex>) { return m_mi->template modify<t_tag>(it, std::forward<P_Fn>(fn)); }
      template<class P_V>
      bool replace(const_iterator it, P_V&& v) requires (!std::is_const_v<P_MultiIndex>) { return m_mi->template replace<t_tag>(it, std::forward<P_V>(v)); }

   private:
      const t_map& GetMap() const { return m_mi->template GetMap<I_Index>(); }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Convert raw iterator to const_iterator, handling materialization if needed
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      const_iterator ToIterator(t_raw_it it) const noexcept {
         if constexpr (c_needs_handle_materialization) {
            if constexpr (I_SkipDead)
               return const_iterator{iterator_base{it, GetMap().end()}};
            else
               return const_iterator{it};
         } else {
            if constexpr (I_SkipDead)
               return const_iterator{it, GetMap().end()};
            else
               return it;
         }
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Helper to get the original raw iterator, somtimes unwrapping twice
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      static inline auto RawIterator(const_iterator it) noexcept {
         if constexpr (c_needs_handle_materialization) {
            if constexpr (I_SkipDead)
               return it.Raw().Raw();
            else
               return it.Raw();
         } else {
            if constexpr (I_SkipDead)
               return it.Raw();
            else
               return it;
         }
      }
   };

   //###############################################################################################################################################
   // Secondary that needs translation 
   //###############################################################################################################################################
   template<class P_MultiIndex, std::size_t I_Index>
   class T_IndexView<P_MultiIndex, I_Index, /*skip*/false, /*translate*/true> {
      template<class P_Key, class P_Payload, bool I_PerThreadErr, template<class> class P_Policy, class... P_Specs>
      friend class T_MultiIndex;

      using t_mi = std::remove_const_t<P_MultiIndex>;
      using t_map = std::decay_t<decltype(std::declval<P_MultiIndex*>()->template GetMap<I_Index>())>;
      using t_policy = typename t_mi::t_policy_base;
      using t_handle = typename t_mi::t_handle;
      using t_tag = typename t_mi::template t_spec<I_Index>::t_tag;

      P_MultiIndex* m_mi;

   public:
      using key_type = typename t_map::key_type;
      using mapped_type = t_handle;
      using const_iterator = T_HandleIter<typename t_map::const_iterator, t_policy, t_handle>;
      using size_type = std::size_t;

      explicit T_IndexView(P_MultiIndex* mi) : m_mi(mi) {}

      const_iterator begin() const { return {GetMap().begin(), GetPolicy()}; }
      const_iterator end()   const { return {GetMap().end(),   GetPolicy()}; }

      const_iterator find(const key_type& key) const {
         auto it = GetMap().find(key);
         return it == GetMap().end() ? end() : const_iterator(it, GetPolicy());
      }

      auto equal_range(const key_type& key) const {
         auto [f,l] = GetMap().equal_range(key);
         return std::make_pair(const_iterator(f, GetPolicy()), const_iterator(l, GetPolicy()));
      }
      bool contains(const key_type& key) const { return find(key) != end(); }

      bool      empty() const { return size() == 0; }
      size_type size()  const { return GetMap().size(); }
      size_type count(const key_type& key) const { return GetMap().count(key); }
      size_type erase(const key_type& key) requires (!std::is_const_v<P_MultiIndex>) { return m_mi->template erase<t_tag>(key); }
      const_iterator erase(const_iterator it) requires (!std::is_const_v<P_MultiIndex>) { return m_mi->template erase<t_tag>(it); }

      template<class P_Fn>
      bool modify(const_iterator it, P_Fn&& fn) requires (!std::is_const_v<P_MultiIndex>) { return m_mi->template modify<t_tag>(it, std::forward<P_Fn>(fn)); }
      template<class P_V>
      bool replace(const_iterator it, P_V&& v) requires (!std::is_const_v<P_MultiIndex>) { return m_mi->template replace<t_tag>(it, std::forward<P_V>(v)); }

   private:
      const t_map& GetMap() const { return m_mi->template GetMap<I_Index>(); }
      const t_policy* GetPolicy() const { return static_cast<const t_policy*>(m_mi); }
      static inline auto RawIterator(const_iterator it) noexcept { return it.Raw(); }
   };

   //###############################################################################################################################################
   // Helper for nothrow swap determination
   //###############################################################################################################################################
   
   template<class P_Map, bool = HasAllocator<P_Map>>
   struct S_MiNothrowSwap : std::true_type { };

   template<class P_Map>
   struct S_MiNothrowSwap<P_Map, true> : std::bool_constant<std::allocator_traits<typename P_Map::allocator_type>::is_always_equal::value> {
   };

   //###############################################################################################################################################
   // T_MultiIndex
   //
   // MAIN CLASS IMPLEMENTATION
   // 
   // Complexity guarantees (typical/amortized):
   // - emplace: O(log N) or O(1) primary + O(#indices) secondaries
   // - modify/replace: O(#indices) for drop+rebuild
   // - erase: O(#indices) for secondary removal
   // - find: O(log N) or O(1) depending on map type
   // - Relocation cost varies by policy (see policy comments)
   // 
   // Memory layout optimizations:
   // - Empty Base Optimization for unused metadata
   // - Conditional inclusion of live counter, owner pointer, dead flag
   // - Translation array uses rebound allocator when available
   //
   //###############################################################################################################################################
   template<class P_Key, class P_Payload, bool I_PerThreadErr, template<class> class P_Policy, class... P_Specs>
   class T_MultiIndex : private T_LiveCounterBase<P_Policy<T_MultiIndex<P_Key, P_Payload, I_PerThreadErr, P_Policy, P_Specs...>>::c_uses_tombstones, I_PerThreadErr>,
      public P_Policy<T_MultiIndex<P_Key, P_Payload, I_PerThreadErr, P_Policy, P_Specs...>>,
      public S_AllocFacadeBase<
                                 typename std::tuple_element_t<stor_pos_v<P_Specs...>, std::tuple<P_Specs...> >::template t_map_type<
                                    P_Key,
                                    T_PayloadWrap<
                                       P_Payload,
                                       P_Policy<T_MultiIndex<P_Key, P_Payload, I_PerThreadErr, P_Policy, P_Specs...>>::c_uses_tombstones,
                                       P_Policy<T_MultiIndex<P_Key, P_Payload, I_PerThreadErr, P_Policy, P_Specs...>>::c_invalidates,
                                       P_Policy<T_MultiIndex<P_Key, P_Payload, I_PerThreadErr, P_Policy, P_Specs...>>::c_needs_translation_array,
                                       P_Policy<T_MultiIndex<P_Key, P_Payload, I_PerThreadErr, P_Policy, P_Specs...>>
                                    >
                                 >
                              > {
      using t_self = T_MultiIndex<P_Key, P_Payload, I_PerThreadErr, P_Policy, P_Specs...>;
      using t_policy_base = P_Policy<t_self>;
      friend t_policy_base;
      template<class P_MultiIndex, std::size_t I_Index, bool I_SkipDead, bool I_Translate>
      friend class T_IndexView;

      // compile-time facts
      constexpr static std::size_t prim_cnt = primary_count<P_Specs...>;
      static_assert(prim_cnt == 1, "Exactly one index<> must be Storage=true");
      static constexpr std::size_t c_prim_idx = stor_pos_v<P_Specs...>;
      using t_prim_spec = std::tuple_element_t<c_prim_idx, std::tuple<P_Specs...>>;
      static constexpr bool c_invalidates = t_policy_base::c_invalidates;
      static constexpr bool c_tombstones = t_policy_base::c_uses_tombstones;
      static constexpr bool c_needs_transl_array = t_policy_base::c_needs_translation_array;

      using t_live_base = T_LiveCounterBase<c_tombstones, I_PerThreadErr>;

      // fundamental types
      using t_prim_tag = typename t_prim_spec::t_tag;
      using t_wrap = T_PayloadWrap<P_Payload, c_tombstones, c_invalidates, c_needs_transl_array, t_policy_base>;
      using t_prim_map = typename t_prim_spec::template t_map_type<P_Key, t_wrap>;
      using t_pair = typename t_prim_map::value_type;
      using t_handle = T_Handle<t_pair>;
      static constexpr bool c_primary_is_unique = is_unique_v<t_prim_map>;


      // Design: Secondary map type depends on policy's storage strategy
      template<class P_S> struct S_SecMap {
         using t_key = typename P_S::template t_key_type<P_Key, P_Payload>;
         using type = std::conditional_t<
                         P_S::is_storage,
                         std::monostate,
                         std::conditional_t<
                            t_policy_base::c_stores_handle,
                            typename P_S::template t_map_type<t_key, t_handle>,
                            std::conditional_t<
                               t_policy_base::c_needs_translation_array,
                               typename P_S::template t_map_type<t_key, std::size_t>,
                               typename P_S::template t_map_type<t_key, P_Key>
                            >
                         >
                      >;
      };
      using t_sec_tuple = std::tuple<typename S_SecMap<P_Specs>::type...>;
      // allocator typedefs come from S_AllocFacadeBase when they exist

   public:
      using key_type = P_Key;
      using size_type = std::size_t;
      using value_type = std::pair<const key_type, P_Payload>;
      using t_prim_map_iter = typename t_prim_map::const_iterator;
      using const_iterator = std::conditional_t<c_tombstones, T_LiveIter<t_prim_map_iter>, t_prim_map_iter>;
      using iterator      = const_iterator;
      using difference_type = typename const_iterator::difference_type;

   private:
      t_prim_map m_storage;
      t_sec_tuple m_second;

   public:
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // A helper class that allows a more natural synatx - only on the main class, not the 'view' intervaces.
      // Design: RAII proxy for operator[]. Captures old state, applies changes on destruction.
      // On failure, rolls back silently (sets thread-local/static flag).
      // Prefer explicit commit() in production code for observable success/failure.
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      class C_EditProxy {
         T_MultiIndex* m_self;
         t_handle m_h;
         std::pair<const key_type, P_Payload> m_buf;
         bool m_committed{false};

         ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
         // An optimization here: if the node is dead, no need to restore the old payload, so, no need to have an intermediate copy
         ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
         bool Revive(t_handle h) {
            h.Flag() = false;
            h.Payload() = conditional_move(m_buf.second);
            if constexpr (c_invalidates)
               h.Raw()->second.m_p_owner = static_cast<t_policy_base*>(m_self);
            if (!m_self->AddSecs(h)) {
               h.Flag() = true;
               return false;
            }
            m_self->IncrementLive();
            return true;
         }
      public:
         C_EditProxy(T_MultiIndex* p_s, t_handle h) : m_self(p_s), m_h(h), m_buf(h.Key(), h.Payload()) { }
         C_EditProxy(T_MultiIndex* p_s, const key_type& k) : m_self(p_s), m_buf(k, P_Payload{}) { }
         C_EditProxy(const C_EditProxy&) = delete;
         C_EditProxy& operator=(const C_EditProxy&) = delete;

         operator std::pair<const key_type, P_Payload>&() { return m_buf; }
         auto* operator->() { return &m_buf.second; }
         auto& operator*() { return m_buf.second; }

         bool commit() {
            m_committed = true;
            EditOk() = true;
            if (!m_h)
               return EditOk() = m_self->emplace(m_buf.first, conditional_move(m_buf.second)).second;

            if constexpr (c_tombstones) {
               if (!m_h.Flag())             // live -> replace
                  return EditOk() = m_self->Replace(m_h, conditional_move(m_buf.second));
               return EditOk() = Revive(m_h);  // dead -> revive
            }
            else
               return EditOk() = m_self->Replace(m_h, conditional_move(m_buf.second));
         }
         void abort() { m_committed = true; }
         ~C_EditProxy() { if (!m_committed) try { commit(); } catch (...) { EditOk() = false; } }
      };

   public:
      T_MultiIndex() = default;

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

      template<class P_Alloc>
      explicit T_MultiIndex(const P_Alloc& a) : 
         t_policy_base(a),
         m_storage(MakeMap<t_prim_map>(a)),
         m_second(MakeSecTuple(a)) {
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Copy constructor: Non-invalidating version
      // Can copy storage directly, then rebuild secondaries
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      T_MultiIndex(const T_MultiIndex& rhs) requires (!c_invalidates) : t_live_base(rhs), t_policy_base(rhs), m_storage(rhs.m_storage), 
         m_second([&] {
         if constexpr (HasAllocator<t_prim_map>)
            return MakeSecTuple(rhs.get_allocator());
         else
            return MakeSecTuple();
      }()) {
         for (auto& pr : m_storage)
            AddSecs(t_handle{&pr});
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Copy constructor: Invalidating version
      // Must rebuild element-by-element to establish new node addresses
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      T_MultiIndex(const T_MultiIndex& rhs) requires (c_invalidates) : t_policy_base([&]() {
         if constexpr (HasAllocator<t_prim_map>) 
            return rhs.get_allocator();
         else 
            return std::monostate{};
      }()), //pass allocator to the policy when the primary map has one
         m_storage(), m_second() {
         if constexpr (HasAllocator<t_prim_map>) {
            m_storage = MakeMap<t_prim_map>(rhs.get_allocator());
            m_second = MakeSecTuple(rhs.get_allocator());
         }
         else
            m_second = MakeSecTuple();

         reserve_all(rhs.size());
         for (const auto& pr : rhs.m_storage) {
            if constexpr (c_tombstones) {
               if (!pr.second.IsDead())
                  emplace(pr.first, pr.second.Payload());
            }
            else
               emplace(pr.first, pr.second.Payload());
         }
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Move constructor: Steals resources, then fixes up owner pointers
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      T_MultiIndex(T_MultiIndex&& rhs) : t_live_base(std::move(rhs)), t_policy_base(), m_storage(std::move(rhs.m_storage)), m_second(std::move(rhs.m_second)) { //t_policy_base is reset to default
         rhs.SetLive(0);
         static_cast<t_policy_base&>(*this) = std::move(static_cast<t_policy_base&>(rhs)); //only now steal the policy state (transl, etc.), which has been correctly updated by OnRelocate calls during element-wise moves.
         RebindOwners();
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   private:
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Core copy assignment logic (used by both copy and move assignment)
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      T_MultiIndex& CopyAssignCore(const T_MultiIndex& rhs) {
         if constexpr (c_invalidates) {
            clear();
            reserve_all(rhs.size());
            for (const auto& pr : rhs.m_storage) {
               if constexpr (c_tombstones) {
                  if (!pr.second.IsDead())
                     emplace(pr.first, pr.second.Payload());
               }
               else
                  emplace(pr.first, pr.second.Payload());
            }
            return *this;
         }
         else {
            t_live_base::operator=(rhs); //both live and policy base should be empty, really
            t_policy_base::operator=(rhs);
            m_storage = rhs.m_storage;
            ClearSecondaries();
            for (auto& pr : m_storage)
               AddSecs(t_handle{&pr});
            return *this;
         }
      }

   public:

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Copy assignment: Respects allocator propagation traits
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      T_MultiIndex& operator=(const T_MultiIndex& rhs) {
         if (this == &rhs)
            return *this;

         if constexpr (ReassignAllocatorOnCopy<t_prim_map>()) {
            if (get_allocator() != rhs.get_allocator()) {
               m_storage = MakeMap<t_prim_map>(rhs.get_allocator());
               m_second = MakeSecTuple(rhs.get_allocator());
               t_policy_base::PolicyResetAllocator(rhs.get_allocator());
            }
         }
         return CopyAssignCore(rhs);
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Move assignment: Tries fast move if allocators match, falls back to copy
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      T_MultiIndex& operator=(T_MultiIndex&& rhs) {
         if (this == &rhs) 
            return *this;

         if constexpr (ReassignAllocatorOnMove<t_prim_map>()) {
            if (get_allocator() != rhs.get_allocator()) {
               m_storage = MakeMap<t_prim_map>(rhs.get_allocator());
               m_second = MakeSecTuple(rhs.get_allocator());
               t_policy_base::PolicyResetAllocator(rhs.get_allocator());
            }
         }
         // Fallback to copy if allocators mismatch and propagation disallowed
         if constexpr (FastMoveDisallowed<t_prim_map>())
            if (m_storage.get_allocator() != rhs.m_storage.get_allocator())
               return CopyAssignCore(rhs);

         t_live_base::operator=(std::move(rhs));
         m_storage = std::move(rhs.m_storage); //may trigger OnRelocate calls on rhs policy
         m_second = std::move(rhs.m_second);
         // now steal policy state (translation array already patched, for example)
         t_policy_base::operator=(std::move(rhs));

         rhs.SetLive(0);
         RebindOwners();
         return *this;
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

      auto get_allocator() const noexcept requires HasAllocator<t_prim_map> { return m_storage.get_allocator(); }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // capacity
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      bool empty() const noexcept { return size() == 0; }
      size_type size() const noexcept {
         if constexpr (c_tombstones) {
            return this->GetLive();
         }
         else 
            return m_storage.size();
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // lookup (primary)
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      auto equal_range(const key_type& key) const {
         auto [f, l] = m_storage.equal_range(key);
         return std::make_pair( ToIterator(f), ToIterator(l));
      }

      const_iterator find(const key_type& key) const {
         auto [it, it_l] = m_storage.equal_range(key);

         if constexpr (c_tombstones) {
            if (it == it_l)
               return end();
            if (it->second.IsDead()) {
               if constexpr (!c_primary_is_unique) {
                  // If multimap, search forward for a live one with the same key.
                  it = SkipDead(it, it_l);
                  if (it == it_l)
                     return end();
               }
               else
                  return end(); // If unique map, the key is dead.
            }
            return const_iterator{it, m_storage.end()};
         }
         else
            return it == it_l ? end() : const_iterator{it};
      }

      bool contains(const key_type& key) const { return find(key) != end(); }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // emplacement
      // Design: Strong exception guarantee via careful ordering
      // 1. Try primary insertion
      // 2. Call OnEmplaceSuccess (for translation array)
      // 3. Add secondaries (may fail on unique constraint)
      // 4. On failure: rollback via OnEmplaceFail and primary erase
      // 
      // Publication order for concurrency: Primary first, then secondaries.
      // Brief window where element visible in primary but not secondaries.
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      template<class... P_Args>
      std::pair<const_iterator, bool> emplace(key_type k, P_Args&&... args) {
         t_wrap w{std::forward<P_Args>(args)...};

         typename t_prim_map::iterator it;

         if constexpr (c_primary_is_unique) {
            if constexpr (c_tombstones) {             // revive a dead node, if any
               auto it0 = m_storage.find(k);
               if (it0 != m_storage.end()) {
                  if (it0->second.IsDead()) {

                     t_handle h{&*it0};
                     h.Flag() = false;
                     h.Payload() = conditional_move(w.m_value);

                     if constexpr (c_invalidates)
                        h.Raw()->second.m_p_owner = static_cast<t_policy_base*>(this);
                     try {
                        if (!AddSecs(h)) {
                           h.Flag() = true;
                           return {ToIterator(), false};
                        }
                     }
                     catch(...) {
                        h.Flag() = true;
                        throw;
                     }
                     this->IncrementLive();
                     return {ToIterator(it0), true};
                  } 
                  else
                     return {ToIterator(it0), false};
               }
            }

            bool ok;
            std::tie(it, ok) = m_storage.try_emplace(k, std::move(w));
            if (!ok) 
               return {ToIterator(it), false};
         }
         else
            it = m_storage.insert({k, std::move(w)});

         t_handle h{&*it};
         if constexpr (c_invalidates)
            it->second.m_p_owner = static_cast<t_policy_base*>(this);
         bool called_success = false;
         try {
            t_policy_base::OnEmplaceSuccess(h, &*it);
            called_success = true;
            if (!AddSecs(h)) {
               t_policy_base::OnEmplaceFail();
               m_storage.erase(it);
               return {ToIterator(), false};
            }
         }
         catch(...) {
            if (called_success)
               t_policy_base::OnEmplaceFail();
            m_storage.erase(it);
            throw;
         }
         this->IncrementLive();
         return {ToIterator(it), true};
      }

      // std-style wrappers
      std::pair<const_iterator, bool> insert(const value_type& v) { return emplace(v.first, v.second); }
      std::pair<const_iterator, bool> insert(value_type&& v) { return emplace(std::move(const_cast<key_type&>(v.first)), conditional_move(v.second)); }

      template<class... P_Args>
      std::pair<const_iterator, bool> try_emplace(key_type k, P_Args&&... a) {
         // try_emplace semantics are primarily for unique containers.
         if constexpr (c_primary_is_unique) {
            if (auto it = find(k); it != end())
               return {it, false};
         }
         return emplace(std::move(k), std::forward<P_Args>(a)...);
      }

      template<class P_M>
      std::pair<const_iterator, bool> insert_or_assign(key_type k, P_M&& m) {
         static_assert(c_primary_is_unique, "insert_or_assign requires a unique primary index.");

         if (auto it = find(k); it != end()) {
            UpdateCore(ToHandle(it), [&](P_Payload& dst) { dst = safe_forward<P_M>(m); });
            return {it, false};
         }
         return emplace(std::move(k), std::forward<P_M>(m));
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

      template<class P_V>
      bool replace(const_iterator it, P_V&& v) { 
         if (it==end()) 
            return false; 
         return Replace(ToHandle(it), std::forward<P_V>(v)); 
      }
      template<class P_Tag, class P_Iter, class P_V>
      bool replace(P_Iter view_it, P_V&& v) requires std::same_as<P_Iter, typename decltype(this->template get<P_Tag>())::const_iterator> {
         auto view = get<P_Tag>();
         if (view_it == view.end())
            return false;
         return Replace(view_it->second, std::forward<P_V>(v));
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

      template<class P_Fn>
      bool modify(const_iterator it, P_Fn&& fn) { 
         if (it==end()) 
            return false; 
         return UpdateCore(ToHandle(it), [&](P_Payload& dst) { std::forward<P_Fn>(fn)(dst); }); 
      }
      
      template<class P_Tag, class P_Iter, class P_Fn>
      bool modify(P_Iter view_it, P_Fn&& fn) requires std::same_as<P_Iter, typename decltype(this->template get<P_Tag>())::const_iterator> {
         auto view = get<P_Tag>();
         if (view_it == view.end())
            return false;
         return UpdateCore(view_it->second, [&](P_Payload& dst) { std::forward<P_Fn>(fn)(dst); });
      }

      
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // count / equal_range
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      size_type count(const key_type& k) const {
         if constexpr (!c_primary_is_unique) {
            auto [f, l] = m_storage.equal_range(k);
            if constexpr (c_tombstones) {
               size_type n = 0;
               for (auto it = f; it != l; ++it)
                  if (!it->second.IsDead())
                     ++n;
               return n;
            }
            else
               return static_cast<size_t>(std::distance(f, l));
         }
         else
            return find(k) == end() ? 0 : 1;
      }

      template<class P_Tag>
      size_type count(const typename std::tuple_element_t<tag_pos_v<P_Tag, P_Specs...>, std::tuple<P_Specs...>>::template t_key_type<P_Key, P_Payload>& sk) const {
         return get<P_Tag>().count(sk);
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

      auto bucket_count() const requires requires { m_storage.bucket_count(); } { return m_storage.bucket_count(); }
      void reserve(size_type n) requires requires { m_storage.reserve(n); } { m_storage.reserve(n); }
      void rehash(size_type n) requires requires { m_storage.rehash(n); } { m_storage.rehash(n); }
      float load_factor() const requires requires { m_storage.load_factor(); } { return m_storage.load_factor(); }
      void max_load_factor(float f) requires requires { m_storage.max_load_factor(f); } { m_storage.max_load_factor(f); }

      void reserve_all(size_type n) {
         if constexpr (requires { reserve(n); })
            reserve(n);
         ForEachSecondary([&]<std::size_t I_I>() -> bool {
            if constexpr (!t_spec<I_I>::is_storage)
               if constexpr (requires { idx<I_I>().reserve(n); })
                  idx<I_I>().reserve(n);
            return true;
         });
      }

      void swap(T_MultiIndex& o) noexcept(nothrow_swap) {
         using std::swap;
         swap(m_storage, o.m_storage);
         swap(m_second, o.m_second);
         t_policy_base::PolicySwap(static_cast<t_policy_base&>(o));
         this->SwapLive(o);
         RebindOwners();
         o.RebindOwners();
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // compact gaps (O(N) copy-swap)
      // Design: Removes tombstones or re-densifies translation array
      // O(N) rebuild via temporary container and swap
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      void compact() requires (c_tombstones || c_needs_transl_array) {
         auto make_tmp = [this]() {
            if constexpr (HasAllocator<t_prim_map>)
               return T_MultiIndex(get_allocator());
            else
               return T_MultiIndex();
         };

         auto tmp = make_tmp();
         tmp.reserve_all(size());

         for (const auto& pr : m_storage) {
            if constexpr (c_tombstones) {
               if (!pr.second.IsDead())
                  tmp.emplace(pr.first, pr.second.Payload());
            } else {
               // If c_needs_transl_array=true but c_tombstones=false, all are live.
               tmp.emplace(pr.first, pr.second.Payload());
            }
         }
         swap(tmp);
      }

      C_EditProxy operator[](const key_type& k) requires(c_primary_is_unique) {
         auto it = m_storage.find(k);
         if (it != m_storage.end())
            return C_EditProxy{this, t_handle{&*it}}; // live or dead
         return C_EditProxy{this, k};                  // new key
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Erase by primary key
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      size_type erase(const key_type& key) {
         auto [it, it_l] = equal_range(key);
         if (it == it_l)
            return 0;

         size_type removed = 0;
         for (;;) {
            it = erase(it);
            ++removed;

            if constexpr (!c_primary_is_unique) {
               if (it == it_l)
                  break;
            }
            else
               break;
         }
         return removed;
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Erase by secondary key
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      template<class P_Tag>
      size_t erase(const typename std::tuple_element_t< tag_pos_v<P_Tag, P_Specs...>, std::tuple<P_Specs...>>::template t_key_type<P_Key, P_Payload>& skey) {
         constexpr std::size_t I = tag_pos_v<P_Tag, P_Specs...>;
         if constexpr (I != c_prim_idx) {
            auto& m = idx<I>();
            size_type removed = 0;
            auto view_it = m.find(skey);
            if (view_it == m.end())
               return 0;
            for (;;) {
               t_handle h = t_policy_base::template ToHandle<t_handle>(view_it->second);
               EraseByHandle<I>(h); //dropping everything except for the current secondary index
               ++removed;
               view_it = m.erase(view_it);
               if constexpr (!is_unique_v<decltype(m)>) {
                  if (view_it == m.end() || view_it->first!=skey)
                     break;
               }
               else
                  break;
            }
            return removed;
         }
         else
            return erase(skey);
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Erase by secondary iterator
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      template<class P_Tag, class P_Iter>
      auto erase(P_Iter view_it) requires std::same_as<P_Iter, typename decltype(this->template get<P_Tag>())::const_iterator> {
         using t_it = decltype(view_it);

         constexpr std::size_t I = tag_pos_v<P_Tag, P_Specs...>;
         auto raw_view_it = get<P_Tag>().RawIterator(view_it);
         if constexpr (I != c_prim_idx) {
            auto& m = idx<I>();

            if (raw_view_it == m.end())
               return view_it;

            t_handle h = t_policy_base::template ToHandle<t_handle>(raw_view_it->second);
            EraseByHandle<I>(h); //dropping everything except for the current secondary index
            if constexpr (requires {t_it(m.erase(raw_view_it));})
               return t_it(m.erase(raw_view_it));
            else
               return t_it(m.erase(raw_view_it), static_cast<const t_policy_base*>(this));
         }
         else
            return t_it(erase(raw_view_it));
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Erase by primary (T_MultiIndex) iterator
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      const_iterator erase(const_iterator it) {
         if (it == end())
            return end();

         auto h = ToHandle(it);
         if (!h)
            return it;
         DropSecs(h);
         if constexpr (c_tombstones) {
            h.Flag() = true;
            this->DecrementLive();
            return ++it;
         }
         else {
            const_iterator ret_it = const_iterator(m_storage.erase(it));
            this->DecrementLive();
            return ret_it;
         }
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Erase by primary() storage iterator
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      const_iterator erase(t_prim_map_iter raw_it) requires(c_tombstones) {
         T_LiveIter<t_prim_map_iter> wrap{raw_it, m_storage.end()};
         return erase(wrap);
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

      void clear() {
         m_storage.clear();
         ClearSecondaries();
         t_policy_base::PolicyClear();
         this->SetLive(0);
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // iteration
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      iterator begin() const noexcept { return ToIterator(m_storage.begin()); }
      iterator end() const noexcept { return ToIterator(m_storage.end()); }
      const_iterator cbegin() const noexcept { return begin(); }
      const_iterator cend()   const noexcept { return end();   }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // primary / views
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      const auto& primary() const noexcept { return m_storage; }

      template<class P_Tag>
      auto get() noexcept { return Get<P_Tag>(this); }

      template<class P_Tag>
      auto get() const noexcept { return Get<P_Tag>(this); }

   private:
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Handle <-> Iterator helpers
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      [[nodiscard]] t_handle ToHandle(const_iterator it) const noexcept{
         if (it == end())
            return t_handle{};
         if constexpr (c_tombstones)
            return t_handle{ const_cast<t_handle::t_pair*>(&*it.Raw()) };
         else
            return t_handle{ const_cast<t_handle::t_pair*>(&*it) };
      }

      [[nodiscard]] const_iterator ToIterator(typename t_prim_map::iterator it) const noexcept {
         if constexpr (c_tombstones)
            return const_iterator{it, m_storage.end()};
         else
            return const_iterator{it};
      }
      [[nodiscard]] const_iterator ToIterator(typename t_prim_map::const_iterator it) const noexcept {
         if constexpr (c_tombstones)
            return const_iterator{it, m_storage.end()};
         else
            return const_iterator{it};
      }
      [[nodiscard]] const_iterator ToIterator()  const noexcept {
         if constexpr (c_tombstones)
            return const_iterator{m_storage.end(), m_storage.end()};
         else
            return const_iterator{m_storage.end()};
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Single template implementation that handles both const and non-const
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      template<class P_Tag, class P_Self>
      static auto Get(P_Self* self) noexcept {
         constexpr std::size_t I_Idx = tag_pos_v<P_Tag, P_Specs...>;
         if constexpr (I_Idx != c_prim_idx)
            return T_IndexView<P_Self, I_Idx, /*skip*/false, /*translate*/!t_policy_base::c_stores_handle>(self);
         else
            return T_IndexView<P_Self, I_Idx, c_tombstones, /*translate*/false>(self);
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Helper methods to access maps by index (used by T_IndexView)
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      template<std::size_t I>
      auto& GetMap() noexcept { if constexpr (I == c_prim_idx) return m_storage; else return idx<I>(); }
      template<std::size_t I>
      const auto& GetMap() const noexcept { if constexpr (I == c_prim_idx) return m_storage; else return idx<I>(); }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Owner pointer rebinding after move/swap
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      void RebindOwners() {
         if constexpr (c_invalidates) {
            for (auto& pr : m_storage)
               pr.second.m_p_owner = static_cast<t_policy_base*>(this);
         }
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Unified helper for secondary operations using fold expression
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      template<class P_F>
      static constexpr auto ForEachSecondary(P_F&& f) {
         return [&]<std::size_t... I_Is>(std::index_sequence<I_Is...>) {
            return (f.template operator()<I_Is>() && ...);
         }(std::make_index_sequence<sizeof...(P_Specs)>{});
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Helper to clear secondary maps (used by clear() and assignment)
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      void ClearSecondaries() {
         ForEachSecondary([&]<std::size_t I_I>() -> bool {
            if constexpr (!t_spec<I_I>::is_storage) {
               idx<I_I>().clear();
            }
            return true;
         });
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Thread-local or static flag for EditProxy success tracking
      // Design: I_PerThreadErr=true uses thread_local for multi-threaded read scenarios
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      static inline bool& EditOk() noexcept {
         if constexpr (I_PerThreadErr) {
            thread_local bool ok = true;
            return ok;
         }
         else {
            static bool ok = true;
            return ok;
         }
      }
      
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // WARNING: container_of pattern - assumes std::pair is standard-layout
      // Works in practice with major implementations but technically not guaranteed
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      static t_pair* WrapToPair(t_wrap* p_w) noexcept { 
         #if defined(__clang__)
         #pragma clang diagnostic push
         #pragma clang diagnostic ignored "-Winvalid-offsetof"
         #elif defined(__GNUC__)
         #pragma GCC diagnostic push
         #pragma GCC diagnostic ignored "-Winvalid-offsetof"
         #endif
            return reinterpret_cast<t_pair*>(reinterpret_cast<char*>(p_w) - offsetof(t_pair, second));
         #if defined(__clang__)
         #pragma clang diagnostic pop
         #elif defined(__GNUC__)
         #pragma GCC diagnostic pop
         #endif
      }
      static const t_pair* WrapToPair(const t_wrap* p_w) noexcept { 
         #if defined(__clang__)
         #pragma clang diagnostic push
         #pragma clang diagnostic ignored "-Winvalid-offsetof"
         #elif defined(__GNUC__)
         #pragma GCC diagnostic push
         #pragma GCC diagnostic ignored "-Winvalid-offsetof"
         #endif
            return reinterpret_cast<t_pair*>(reinterpret_cast<char*>(p_w) - offsetof(t_pair, second));
         #if defined(__clang__)
         #pragma clang diagnostic pop
         #elif defined(__GNUC__)
         #pragma GCC diagnostic pop
         #endif
      }

      template<std::size_t I_I>
      using t_spec = std::tuple_element_t<I_I, std::tuple<P_Specs...>>;
      using t_map_types = std::tuple<t_prim_map, typename S_SecMap<P_Specs>::type...>;

      template<std::size_t... I_Is>
      static consteval bool MapsNothrow(std::index_sequence<I_Is...>) {
         return (S_MiNothrowSwap<std::tuple_element_t<I_Is, t_map_types>>::value && ...);
      }
      static constexpr bool nothrow_swap = MapsNothrow(std::make_index_sequence<std::tuple_size_v<t_map_types>>{});
      template<std::size_t I_I> auto& idx() { return std::get<I_I>(m_second); }
      template<std::size_t I_I> const auto& idx() const { return std::get<I_I>(m_second); }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // allocator-aware helper
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      template<std::size_t I_I>
      static auto MakeSecElem(const auto& a) {
         using t_map = typename S_SecMap<t_spec<I_I>>::type;
         if constexpr (t_spec<I_I>::is_storage)
            return t_map{};
         else
            return MakeMap<t_map>(a);
      }
      template<std::size_t... I_Is>
      static t_sec_tuple MakeSecTupleImpl(const auto& a, std::index_sequence<I_Is...>) { return t_sec_tuple{MakeSecElem<I_Is>(a)...}; }
      static t_sec_tuple MakeSecTuple(const auto& a) { return MakeSecTupleImpl(a, std::make_index_sequence<sizeof...(P_Specs)>{}); }
      static t_sec_tuple MakeSecTuple() { return MakeSecTupleImpl(std::monostate{}, std::make_index_sequence<sizeof...(P_Specs)>{}); }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Rolling back secondary indices when an insertion fails
      // Design: On failure at secondary I, remove from secondaries 0..I-1
      // Maintains strong exception guarantee for emplace
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      template<std::size_t I_I>
      void Rollback(t_handle h) {
         ForEachSecondary([&]<std::size_t I_J>() -> bool {
            if constexpr (I_J < I_I && !t_spec<I_J>::is_storage) {
               auto& m = idx<I_J>();
               auto k = t_spec<I_J>::Project(h.Key(), h.Payload());
               auto [f, l] = m.equal_range(k);
               for (auto it = f; it != l; ++it)
                  if (t_policy_base::MatchSecondary(it, h)) {
                     m.erase(it);
                     break;
                  }
            }
            return true;
         });
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Adding secondary indices
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      bool AddSecs(t_handle h) {
         try {
            return ForEachSecondary([&]<std::size_t I_I>() -> bool {
               if constexpr (!t_spec<I_I>::is_storage) {
                  auto& m = idx<I_I>();
                  auto k = t_spec<I_I>::Project(h.Key(), h.Payload());
                  if constexpr (is_unique_v<decltype(m)>) {
                     auto [_, ok] = m.insert({k, t_policy_base::GetSecondaryValue(h)});
                     if (!ok) {
                        Rollback<I_I>(h);
                        return false;
                     }
                  }
                  else
                     m.insert({k, t_policy_base::GetSecondaryValue(h)});
               }
               return true;
            });
         }
         catch(...) {
            DropSecs(h);
            throw;
         }
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Removing secondary indices. I_Ignore is important - if that index is being dropped, we have to do it separately to get the next iterator.
      // Design: DropSecs must not throw for exception safety guarantees
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      template<std::size_t I_Ignore = size_t(-1)>
      void DropSecs(t_handle h) {
         ForEachSecondary([&]<std::size_t I_I>() -> bool {
            if constexpr (!t_spec<I_I>::is_storage && I_I!=I_Ignore) {
               auto& m = idx<I_I>();
               auto k = t_spec<I_I>::Project(h.Key(), h.Payload());
               auto [f, l] = m.equal_range(k);
               for (auto it = f; it != l; ++it) {
                  if (t_policy_base::MatchSecondary(it, h)) {
                     m.erase(it);
                     break;
                  }
               }
            }
            return true;
         });
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Common logic for update operations
      // Design: "Drop and rebuild" pattern for secondaries
      // Not minimal-delta but provides strong exception guarantee
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      template<class P_MutFn>
      bool UpdateCore(t_handle h, P_MutFn&& mut) {
         if (!h) 
            return false;
         P_Payload old = h.Payload();
         bool old_flag = false;
         if constexpr (c_tombstones) {
            old_flag = h.Flag();
            h.Flag() = false;
         }

         // Only drop secondaries if the element was previously live (or if c_tombstones are disabled).
         // If DropSecs throws, nothing we can really do - it is not atomic, unknown which index threw, the state will be inconsistent.
         // DropSecs really shouldn't throw to provide any exception safety guarantees.
         if constexpr (c_tombstones) {
            if (!old_flag) 
               DropSecs(h);
         } else
            DropSecs(h);

         auto restore_old =[&]() {
            h.Payload() = conditional_move(old);
            if constexpr (c_tombstones) {
               h.Flag() = old_flag;
               if (!old_flag)
                  AddSecs(h);
            } 
            else
               AddSecs(h);
         };

         try { std::forward<P_MutFn>(mut)(h.Payload()); }
         catch (...) {
            restore_old(); //If it throws, the program will terminate, but nothing we can do about that
            throw;
         }

         if (!AddSecs(h)) {
            restore_old();
            return false;
         }
         if constexpr (c_tombstones)
            if (old_flag)
               this->IncrementLive(); //we revived the dead node
         return true;
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Helper for erase operations. I_Ignore is important - if that index is being dropped, we have to do it separately to get the next iterator.
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      template<std::size_t I_Ignore = size_t(-1)>
      void EraseByHandle(t_handle h) {
         if (!h) 
            return;
         DropSecs<I_Ignore>(h);
         if constexpr (c_tombstones) {
            h.Flag() = true;
            this->DecrementLive();
         }
         else {
            auto [pf, pl] = m_storage.equal_range(h.Key());
            for (auto p = pf; p != pl; ++p) {
               if (&p->second == &h.Raw()->second) {
                  m_storage.erase(p);
                  this->DecrementLive();
                  break;
               }
            }
         }
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Optimization: Skip if value unchanged
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      template<class P_V>
      bool Replace(t_handle h, P_V&& v) {
         if (!h) 
            return false;

         if constexpr (O_EqualityComparable<P_Payload> && std::is_same_v<std::remove_cvref_t<P_V>, P_Payload>) {
            bool is_live = true;
            if constexpr (c_tombstones)
               is_live = !h.Flag();
            if (is_live && h.Payload() == v)
               return true; // no change
         }
         return UpdateCore(h, [nv = safe_forward<P_V>(v)](P_Payload& dst) mutable { dst = conditional_move(nv); });
      }
   };

   // CTAD
   template<class P_K, class P_P, class... P_S>
   T_MultiIndex(std::type_identity<P_K>, std::type_identity<P_P>, P_S...) -> T_MultiIndex<P_K, P_P, /*PerThreadErr=*/false, S_UpdatePointerPolicy, P_S...>;

} // namespace ns_mi
#endif