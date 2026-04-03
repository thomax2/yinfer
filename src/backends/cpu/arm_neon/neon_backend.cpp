#include "neon_ops.h"
#include <unordered_map>
#include <string>
#include "llm_engine/memory/memory_pool.h"
#include "kernel_common.h"

namespace llm_engine {

using MatmulFunc =
    Status(*)(const Tensor&,const Tensor&,Tensor&);

class BackendRegistry {

public:
    static BackendRegistry& instance() {
        static BackendRegistry r;
        return r;
    }

    void register_matmul( const std::string& backend, MatmulFunc f ) {
        matmul_map[backend] = f;
    }

    MatmulFunc get( const std::string& backend ) {
        return matmul_map[backend];
    }

private:
    std::unordered_map<std::string,MatmulFunc> matmul_map;

};

static void register_neon()
{
    auto matmul_neon_auto_workspace = [](
        const Tensor& A,
        const Tensor& B,
        Tensor& C
    ) -> Status {
        if (g_memory_pool == nullptr) {
            return Status::OUT_OF_MEMORY;
        }

        int M = A.shape[0];
        int K = A.shape[1];
        int N = B.shape[1];
        int mp = (M + arm_neon::MR - 1) / arm_neon::MR;
        int np = (N + arm_neon::NR - 1) / arm_neon::NR;

        size_t ws_size = static_cast<size_t>(mp) * arm_neon::MR * K * sizeof(float)
                   + static_cast<size_t>(np) * arm_neon::NR * K * sizeof(float);

        float* workspace = static_cast<float*>(g_memory_pool->allocate(ws_size));
        if (workspace == nullptr) {
            return Status::OUT_OF_MEMORY;
        }

        Status status = arm_neon::matmul_neon(A, B, C, workspace, false, nullptr);
        g_memory_pool->free_block(workspace);
        return status;
    };

    BackendRegistry::instance()
        .register_matmul(
            "CPU_NEON",
            matmul_neon_auto_workspace
        );
}

static int neon_reg = (register_neon(),0);

}