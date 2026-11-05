#include "llama.h"
#include "llama-ext.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static std::vector<float> decode(llama_context * ctx, std::vector<llama_token> tokens) {
    auto batch = llama_batch_get_one(tokens.data(), int32_t(tokens.size()));
    require(llama_decode(ctx, batch) == 0, "decode");
    llama_synchronize(ctx);
    const float * logits = llama_get_logits_ith(ctx, -1);
    require(logits != nullptr, "logits available");
    return {logits, logits + 16};
}

static void equal(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size() == b.size(), "logit size");
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]) && std::fabs(a[i] - b[i]) < 1e-6f, "logit equivalence");
    }
}

int main(int argc, char ** argv) {
    require(argc == 2, "fixture model argument");
    llama_backend_init();
    require(!llama_memory_has_partial_seq_rm(nullptr), "null is not certified");
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_extra_bufts = false;
    auto * model = llama_model_load_from_file(argv[1], mp);
    require(model != nullptr, "model loads");
    auto cp = llama_context_default_params();
    cp.n_ctx = 64;
    cp.n_batch = cp.n_ubatch = 8;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.n_seq_max = 1;
    auto * reference = llama_init_from_model(model, cp);
    auto * optimized = llama_init_from_model(model, cp);
    require(reference && optimized, "contexts initialize");
    auto * mr = llama_get_memory(reference);
    auto * mo = llama_get_memory(optimized);
    require(llama_memory_has_partial_seq_rm(mo), "cache has exact capability");
    decode(reference, {0, 0});
    require(llama_memory_seq_rm(mr, 0, 1, -1), "decoded two-token reference permits partial removal");
    llama_memory_clear(mr, true);
    llama_memory_clear(mo, true);
    equal(decode(reference, {1, 2, 3}), decode(optimized, {1, 2, 3}));
    require(llama_memory_seq_rm(mr, 0, 2, -1), "reference suffix removal");
    require(llama_memory_seq_rm(mo, 0, 2, -1), "optimized suffix removal");
    equal(decode(reference, {4}), decode(optimized, {4}));
    require(llama_memory_seq_rm(mo, 0, 0, 1), "prefix removal");
    require(llama_memory_seq_rm(mo, 0, 100, 101), "absent range removal");
    llama_memory_clear(mo, true);
    require(llama_memory_seq_pos_max(mo, 0) < 0, "cache clears");
    llama_free(reference);
    llama_free(optimized);
    llama_model_free(model);
    llama_backend_free();
    std::puts("PASS: capability, decoded removal, cache state and output logits agree");
}
