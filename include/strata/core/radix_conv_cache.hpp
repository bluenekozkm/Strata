// include/strata/core/radix_conv_cache.hpp - Host-side multi-branch Radix Prefix Tree.
// Manages conversational prefixes on Host RAM with CAS block hashing,
// multi-system-prompt retention, thread-safe prefix matching, and LRU eviction.
#pragma once

#include "strata/core/kv_block.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::core {

/// Forward declarations
struct RadixKvNode;

/// Configuration for RadixConvCache.
struct RadixConvCacheConfig {
    size_t host_ram_budget_bytes = 2ULL * 1024 * 1024 * 1024; // 2 GiB host RAM budget
    size_t max_nodes = 0;                                     // 0 = unlimited
    int format = kKvInt8;                                     // KV tensor format for CAS block hashing
    uint32_t num_layers = 1;
    uint32_t num_heads = 2;
    uint32_t head_dim = 256;
    bool auto_evict_on_insert = true;                         // Trigger eviction if budget exceeded on insert
};

/// Cumulative runtime statistics for RadixConvCache.
struct RadixConvCacheStats {
    uint64_t total_nodes = 0;
    uint64_t total_ram_bytes = 0;
    uint64_t system_prompt_nodes = 0;
    uint64_t pinned_nodes = 0;
    uint64_t prefix_lookups = 0;
    uint64_t prefix_hits = 0;
    uint64_t prefix_misses = 0;
    uint64_t evictions = 0;
    uint64_t bytes_evicted = 0;

    bool operator==(const RadixConvCacheStats& o) const = default;
};

/// Result of matching a token sequence against the Radix Prefix Tree.
struct PrefixMatchResult {
    std::shared_ptr<RadixKvNode> deepest_node = nullptr;   // Deepest node reached in matching
    std::shared_ptr<RadixKvNode> last_full_node = nullptr;  // Deepest node whose tokens were 100% matched
    size_t matched_tokens = 0;                              // Total tokens matched from start of query
    bool is_exact = false;                                  // True if query was completely matched
};

/// Node in the Radix Prefix Tree.
struct RadixKvNode : public std::enable_shared_from_this<RadixKvNode> {
    uint64_t node_id = 0;
    std::weak_ptr<RadixKvNode> parent;
    std::vector<std::shared_ptr<RadixKvNode>> children;
    std::vector<int32_t> tokens;
    uint64_t block_hash = 0;
    std::atomic<bool> is_system_prompt{false};
    std::atomic<bool> is_pinned{false};
    std::atomic<uint64_t> last_accessed_us{0};
    size_t ram_bytes = 0;
    std::shared_ptr<KvBlock> kv_block = nullptr;

    RadixKvNode() = default;
    ~RadixKvNode() = default;

    RadixKvNode(const RadixKvNode&) = delete;
    RadixKvNode& operator=(const RadixKvNode&) = delete;
    RadixKvNode(RadixKvNode&&) = delete;
    RadixKvNode& operator=(RadixKvNode&&) = delete;

    bool is_leaf() const noexcept {
        return children.empty();
    }

    bool is_root() const noexcept {
        return parent.expired();
    }

    std::shared_ptr<RadixKvNode> get_parent() const noexcept {
        return parent.lock();
    }

    size_t compute_ram_bytes() const noexcept;
};

/// Host-Side Multi-Branch Radix Prefix Tree for KV cache management.
class RadixConvCache {
public:
    RadixConvCache();
    explicit RadixConvCache(RadixConvCacheConfig config);
    ~RadixConvCache();

    RadixConvCache(const RadixConvCache&) = delete;
    RadixConvCache& operator=(const RadixConvCache&) = delete;
    RadixConvCache(RadixConvCache&&) = delete;
    RadixConvCache& operator=(RadixConvCache&&) = delete;

    // ---- Prefix Matching (Thread-Safe Reads) ----

    /// Traverse the Radix Tree and find the longest common prefix for the given tokens.
    PrefixMatchResult match_longest_prefix(const std::vector<int32_t>& tokens) const;
    PrefixMatchResult match_longest_prefix(const int32_t* tokens, size_t num_tokens) const;

    // ---- Sequence Insertion (Tree Mutation) ----

    /// Insert a token sequence into the Radix Tree, splitting nodes if divergence occurs.
    std::shared_ptr<RadixKvNode> insert_sequence(
        const std::vector<int32_t>& tokens,
        bool is_system_prompt = false,
        bool is_pinned = false,
        std::shared_ptr<KvBlock> block = nullptr);

    std::shared_ptr<RadixKvNode> insert_sequence(
        const int32_t* tokens,
        size_t num_tokens,
        bool is_system_prompt = false,
        bool is_pinned = false,
        std::shared_ptr<KvBlock> block = nullptr);

    // ---- Pinning & System Prompt Marking ----

    bool mark_system_prompt(uint64_t node_id, bool is_system = true);
    bool mark_system_prompt(const std::shared_ptr<RadixKvNode>& node, bool is_system = true);

    bool pin_node(uint64_t node_id, bool pinned = true);
    bool pin_node(const std::shared_ptr<RadixKvNode>& node, bool pinned = true);

    bool unpin_node(uint64_t node_id);
    bool unpin_node(const std::shared_ptr<RadixKvNode>& node);

    bool is_pinned(uint64_t node_id) const;
    bool is_system_prompt(uint64_t node_id) const;

    // ---- LRU Eviction & Budget Enforcement ----

    /// Trim least recently used unpinned leaf nodes when host memory budget is exceeded.
    /// Pinned nodes and active system prompts are strictly protected.
    size_t evict_to_budget(size_t additional_bytes = 0);

    /// Evict at least bytes_to_free bytes of unpinned leaf nodes if available.
    size_t evict_bytes(size_t bytes_to_free);

    void set_host_ram_budget_bytes(size_t bytes);
    void set_max_nodes(size_t max_nodes);

    // ---- Node & Branch Retrieval ----

    std::shared_ptr<RadixKvNode> find_node(uint64_t node_id) const;
    std::shared_ptr<RadixKvNode> root() const noexcept { return root_; }

    std::vector<int32_t> get_sequence_tokens(uint64_t node_id) const;
    std::vector<int32_t> get_sequence_tokens(const std::shared_ptr<RadixKvNode>& node) const;

    std::vector<std::shared_ptr<RadixKvNode>> get_branch_nodes(uint64_t node_id) const;
    std::vector<std::shared_ptr<RadixKvNode>> get_branch_nodes(const std::shared_ptr<RadixKvNode>& node) const;

    std::vector<uint64_t> get_branch_block_hashes(uint64_t node_id) const;
    std::vector<uint64_t> get_branch_block_hashes(const std::shared_ptr<RadixKvNode>& node) const;

    std::vector<std::shared_ptr<KvBlock>> get_branch_blocks(uint64_t node_id) const;
    std::vector<std::shared_ptr<KvBlock>> get_branch_blocks(const std::shared_ptr<RadixKvNode>& node) const;

    // ---- KV Block Storage ----

    bool set_kv_block(uint64_t node_id, std::shared_ptr<KvBlock> block);
    std::shared_ptr<KvBlock> get_kv_block(uint64_t node_id) const;

    // ---- Node Removal & Reset ----

    bool remove_node(uint64_t node_id, bool force = false);
    void clear();

    // ---- Statistics & Configuration ----

    RadixConvCacheStats get_stats() const;
    void reset_stats();
    const RadixConvCacheConfig& config() const noexcept { return config_; }
    size_t total_ram_bytes() const;
    size_t total_nodes() const;

private:
    void init_root();
    size_t evict_to_budget_locked(size_t additional_bytes);
    void update_subtree_hashes_locked(const std::shared_ptr<RadixKvNode>& node);
    std::vector<std::shared_ptr<RadixKvNode>> get_branch_nodes_locked(
        const std::shared_ptr<RadixKvNode>& node) const;

    RadixConvCacheConfig config_;
    mutable std::shared_mutex rw_mu_;

    std::shared_ptr<RadixKvNode> root_;
    std::unordered_map<uint64_t, std::shared_ptr<RadixKvNode>> node_map_;
    uint64_t next_node_id_ = 1;
    size_t total_ram_bytes_ = 0;

    mutable std::atomic<uint64_t> prefix_lookups_{0};
    mutable std::atomic<uint64_t> prefix_hits_{0};
    mutable std::atomic<uint64_t> prefix_misses_{0};
    uint64_t evictions_ = 0;
    uint64_t bytes_evicted_ = 0;
};

} // namespace strata::core
