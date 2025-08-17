# ns_mi::T_MultiIndex — A Modern, Policy-Based Multi-Index Container for C++20

**Zero-overhead for single-index use. Flexible, high-performance secondary indices when you need them. Bring Your Own Containers.**

`ns_mi::T_MultiIndex` is a header-only C++20 library providing a container adapter that maintains a primary data store and any number of secondary indices with modern, STL-like ergonomics.

It utilizes a policy-based design to adapt to diverse storage needs—from node-stable maps (`std::map`) to contiguous containers that relocate elements (e.g., `boost::flat_map`), or high-performance custom hash tables. You control how indices are maintained, whether to use lazy deletion (tombstones), and it supports payloads with restricted mobility (move-only or copy-only).

## Design Philosophy: Pay Only for What You Use

If you only use a primary index, `T_MultiIndex` imposes zero overhead (no extra memory or indirection) compared to using the underlying container directly. Secondary indices and advanced features (like relocation tracking or tombstones) add cost only when employed.

## Why T_MultiIndex?

While Boost.MultiIndex is a venerable and battle-tested library, `ns_mi::T_MultiIndex` offers advantages for modern C++:

- **Efficiency by Design**: Avoids the inherent indirection costs often found in older designs when only a primary index is needed.

- **Flexibility (BYOC - Bring Your Own Containers)**: Seamlessly mix heterogeneous containers (e.g., a `std::unordered_map` primary with a `std::multimap` secondary).

- **Policy-Driven Optimization**: Fine-tune behavior for relocation handling and deletion strategy based on your specific workload.

- **C++20 Ergonomics**: Clean implementation using concepts, constraints, and modern features for a better API and strong exception safety.

## Key Features

- **Heterogeneous Containers**: Use different map types (hash, tree, flat) for different indices.
- **Flexible Projections**: Define secondary keys using member pointers, 1-argument lambdas (payload), or 2-argument lambdas (key, payload).
- **STL-like Views**: Access indices via `get<Tag>()`, returning a view with standard operations (`find`, `equal_range`, `erase`, `modify`, `replace`).
- **Policy-Based Architecture**: Control trade-offs between relocation speed, access speed, and memory usage.
- **Optional Tombstones**: Lazy deletion for high-churn workloads; reclaim memory via `compact()`.
- **Allocator Awareness**: Careful propagation semantics matching STL conventions.
- **Support for Restricted Types**: Handles copy-only types and, with restrictions, move-only.

## Quick Start

### 1. Single Primary Index (Zero Overhead)

```cpp
#include "MultiIndex.h"
#include <unordered_map>
#include <string>
#include <iostream>

// Define tags for indices (optional but recommended)
namespace tags { struct primary; }

// Define the container: Key=int, Payload=std::string
// Policy=S_NoInv is optimal for node-stable maps like std::unordered_map.
using Container = ns_mi::T_MultiIndex<
    int, std::string, /*PerThreadErr=*/false, ns_mi::S_NoInv,
    ns_mi::t_primary_index<std::unordered_map, tags::primary>
>;

int main() {
    Container c;
    c.emplace(1, "hello");
    c.emplace(2, "world");

    // Iteration feels exactly like a standard map
    for (const auto& [key, payload] : c) {
        // The internal payload wrapper implicitly converts to const Payload&
        std::cout << key << ": " << payload << std::endl;
    }

    auto it = c.find(1);
    if (it != c.end()) {
        // Access the payload directly: it->second behaves like the string
        std::string value = it->second; // Implicit conversion
        std::cout << "Found: " << value << std::endl;
    }
}
```

### 2. Adding Secondary Indices

Define secondary indices using member pointers or lambdas, using the `t_secondary_index` alias.

```cpp
#include <map>
#include <unordered_multimap>

struct Item { std::string name; std::string category; double price; };
namespace tags { struct primary; struct by_name; struct by_category_length; }

// 1-argument lambda projection: depends only on the payload
// Must be 'inline constexpr' for portability as a Non-Type Template Parameter (NTTP)
inline constexpr auto category_length = [](const Item& p) { return p.category.size(); };

using Container = ns_mi::T_MultiIndex<
  int, Item, false, ns_mi::S_UpdatePointerPolicy, // Use UpdatePointerPolicy if primary might relocate
  ns_mi::t_primary_index<std::map, tags::primary>,
  // Secondary index by member pointer
  ns_mi::t_secondary_index<std::multimap, &Item::name, tags::by_name>,
  // Secondary index by 1-arg lambda
  ns_mi::t_secondary_index<std::unordered_multimap, category_length, tags::by_category_length>
>;

int main() {
    Container c;
    c.emplace(1, Item{"Widget", "Hardware", 29.99});
    c.emplace(2, Item{"Gadget", "Software", 49.99});
    c.emplace(3, Item{"Tool",   "Hardware", 19.99});

    // Get views for secondary indices
    auto by_name = c.get<tags::by_name>();
    auto by_len  = c.get<tags::by_category_length>();

    // Find by name
    if (auto it = by_name.find("Widget"); it != by_name.end()) {
        // it->second is a Handle, which provides ergonomic, pair-like access
        // to the primary key (.first) and the payload (.second).
        std::cout << "ID: " << it->second->first
                  << ", Price: " << it->second->second.price << std::endl;
    }

    // Find all items where category length is 8 ("Hardware" or "Software")
    auto [f, l] = by_len.equal_range(8u);
    for (; f != l; ++f) {
        std::cout << f->second->second.name << " has category length 8.\n";
    }
}
```

### 3. Two-Argument Lambda (Key + Payload)

If the secondary key depends on both the primary key and the payload.

```cpp
// 2-argument lambda projection: depends on key (int) and payload (Item)
inline constexpr auto composite_key = [](int id, const Item& p) {
    return p.category + "_" + std::to_string(id);
};

using C2 = ns_mi::T_MultiIndex<
  int, Item, false, ns_mi::S_UpdatePointerPolicy,
  ns_mi::t_primary_index<std::map, struct primary_tag>,
  ns_mi::t_secondary_index<std::multimap, composite_key, struct composite_tag>
>;
// Usage: c.get<composite_tag>().find("Hardware_1");
```

## The Power of Policies

Policies determine how the container manages the relationship between primary storage and secondary indices, particularly when the primary storage relocates elements (e.g., vector-based maps).

### Policy Cheat Sheet

| Policy | Primary Relocates? | Secondaries Store | Relocation Cost | Notes |
|--------|-------------------|-------------------|-----------------|-------|
| `S_NoInv` | No | `T_Handle` | N/A | Best for node-stable primaries (`std::map`, `std::unordered_map`). Fastest option. |
| `S_UpdatePointerPolicy` | Yes | `T_Handle` | O(N_affected); may involve bucket scan. | General use when relocations are infrequent. Patches handles in place. |
| `S_TranslationArrayPolicy` | Yes | Ordinal (`size_t`) | O(1) via central array. | Best for frequent relocations. Adds indirection on access. Not thread-safe. |
| `S_KeyLookupPolicy` | N/A | Primary Key | Primary lookup on access. | Requires a unique primary index. |

### Tombstones (Lazy Deletion)

All policies have a `...Tombs` variant (e.g., `S_UpdatePointerPolicyTombs`).

- **Enabled**: `erase()` is very fast; it marks the node as "dead" but doesn't remove it. Iteration skips dead nodes.
- **Benefit**: Stable iterators/handles even after erasure; optimized for high-churn scenarios.
- **Cleanup**: Memory usage grows until `compact()` is called (an O(N) operation) to physically remove dead nodes.

## API: Ergonomics and Usage

`T_MultiIndex` is designed to feel like using standard STL containers, minimizing exposure to internal mechanisms.

### Primary Iteration Ergonomics

Iterating over the main container yields iterators behaving like the underlying primary map. The internal payload wrapper implicitly converts to `const Payload&`.

```cpp
Container c;
// ...
auto it = c.find(1);

// Assuming Payload is std::string:
std::string value = it->second; // Works via implicit conversion.

// Structured bindings work naturally:
for (const auto& [key, payload] : c) { /* ... */ }
```

### Secondary Views and Handles

Accessing secondary indices is done via `get<Tag>()`, which returns a View. Iterating a secondary view yields pairs of `[SecondaryKey, T_Handle]`.

While `T_Handle` is used internally and returned by secondary iterators, users rarely need to interact with the `T_Handle` API directly. It provides a proxy that mimics a `std::pair<const Key&, const Payload&>`:

```cpp
auto by_name = c.get<tags::by_name>();
auto it = by_name.find("Widget");

if (it != by_name.end()) {
    // Access via the handle proxy (it->second):
    int primary_key = it->second->first;
    const Item& payload = it->second->second;

    // Access members directly
    double price = it->second->second.price;
}
```

### Mutation APIs

`T_MultiIndex` provides several ways to update elements while maintaining the strong exception guarantee (if modification fails, the container remains in its original state).

#### `modify(iterator, functor)`

Atomically updates the payload using a functor.

```cpp
auto it = c.find(1);
bool success = c.modify(it, [](Item& p) {
    p.price *= 1.10; // Increase price by 10%
    p.category = "Updated";
});
```

#### `replace(iterator, new_value)`

Atomically replaces the entire payload.

```cpp
bool success = c.replace(it, Item{"NewName", "NewCat", 10.0});
```

#### `operator[]` (RAII Edit Proxy)

For containers with a unique primary index, `operator[]` returns an RAII proxy for convenient modification or insertion.

```cpp
// Unique primary required
auto proxy = c[42]; // Creates if not exists, or gets existing
proxy->name = "The Answer";
proxy->price = 42.0;

// Commit changes explicitly
bool committed = proxy.commit();

// Or let the proxy commit upon destruction (less explicit error handling)
{
    auto proxy = c[102];
    proxy->name = "Auto Commit";
}
// To cancel changes: proxy.abort();
```

## Bringing Your Own Containers (BYOC)

You can use any container that meets a minimal STL-like interface.

### Required Operations

- **Iteration**: `begin()`, `end()`, `const_iterator`, `value_type`.
- **Lookup**: `find(key)`, `equal_range(key)`.
- **Erase**: `erase(iterator)` or `erase(key)`.
- **Size**: `size()`, `empty()`, `clear()`.

### Insertion Requirements

- **Unique Maps**: Must provide `.insert(value)` returning `std::pair<iterator, bool>`. Primary unique maps also require `try_emplace(key, mapped)`.
- **Multi-Maps**: Must provide `.insert(value)`.

### Adapting Containers Lacking `equal_range`

If a container (e.g., some custom hash maps) lacks `equal_range`, you can adapt it using a simple wrapper (demonstrated in `MultiIndexTest.cpp` as `T_AddEqualRange`):

```cpp
template<class K, class V, template<class, class> class Base>
struct AddEqualRange : Base<K,V> {
    using BaseT = Base<K,V>;
    using BaseT::BaseT; // inherit ctors
    using typename BaseT::iterator;

    // Implementation for unique maps (finds 0 or 1 element)
    std::pair<iterator, iterator> equal_range(const K& k) {
        auto it = BaseT::find(k);
        return it == BaseT::end() ? std::pair{it, it} : std::pair{it, std::next(it)};
    }
    // (const version omitted for brevity)
};

// Usage:
// ns_mi::t_primary_index<AddEqualRange<K, V, MyCustomMap>, tags::primary>
```

## Advanced Topics

### Support for Restricted Types (Move-Only, Copy-Only)

`T_MultiIndex` is designed to handle payloads with restricted move/copy semantics.

#### Move-Only Types

Payloads like `std::unique_ptr` (movable but not copyable) are supported across all policies as long as you are not modifying them after insertion (so, no replace/modify calls). This is the result of an exception safety guarantee, unfortunately: either modifications can be exception-safe or move-only.

```cpp
using MoveOnlyContainer = ns_mi::T_MultiIndex<
    int, std::unique_ptr<std::string>, false, ns_mi::S_UpdatePointerPolicy,
    ns_mi::t_primary_index<std::map, tags::primary>
>;
MoveOnlyContainer mc;
mc.emplace(1, std::make_unique<std::string>("hello"));
```

### Thread Safety (Conditional Subset)

A subset of operations can be safely performed concurrently if **ALL** the following conditions are met:

- **Operations**: Only `emplace()` + `find()`/`contains()`. (No concurrent `erase`, `modify`, or iteration).
- **Containers**: All indices must use concurrent, node-stable maps (e.g., PPL `concurrent_unordered_map`).
- **Policies**:
  - ✅ `S_NoInv`, `S_KeyLookupPolicy`.
  - ✅ `S_UpdatePointerPolicy` ONLY if the primary never relocates at runtime (e.g., pre-reserved `flat_map`).
  - ❌ `S_TranslationArrayPolicy` (due to unsynchronized translation vector).
- **Tombstones**: If enabled, use `PerThreadErr=true` (the third template parameter) for an atomic live counter, allowing safe concurrent `size()` queries.

**Visibility Note**: `emplace()` publishes to the primary index first, then secondaries. There is no cross-index linearizability.

## Limitations and Sharp Edges

- **`std::pair` Layout Assumption**: Internal mechanisms rely on `offsetof(std::pair, second)`, assuming `std::pair` is standard-layout. This works reliably across major compilers (MSVC, GCC, Clang) but is not strictly guaranteed by the C++ standard.
- **`modify`/`replace` Rebuilds All**: These operations use a "drop-and-rebuild" strategy for all secondary indices to ensure strong exception safety. It is not a minimal-delta update.
- **Tombstone Memory Usage**: With tombstones enabled, physical memory usage grows until `compact()` is called.

## Build and Run Tests

The library is header-only. Simply include `MultiIndex.h`.

### Requirements

- C++20 compiler (MSVC 2022+, Clang 15+, GCC 12+ recommended).

### Running the Test Suite (MultiIndexTest.cpp)

The repository includes a comprehensive test suite.

**Linux / macOS (Clang/GCC):**

```bash
# Basic compilation and execution
g++ -std=c++20 -O2 -pthread MultiIndexTest.cpp -o mi_tests
./mi_tests

# To enable tests using Boost.Container (if installed and BOOST_AVAILABLE is defined in the test file)
g++ -std=c++20 -O2 -pthread -DBOOST_AVAILABLE MultiIndexTest.cpp -o mi_tests_boost
```

**Windows (MSVC):**

```bash
cl /std:c++20 /O2 MultiIndexTest.cpp
MultiIndexTest.exe
```

## License

This project is licensed under MIT-0 (see SPDX headers in the source files). You are free to use it without attribution.