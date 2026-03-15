#include "neon_ops.h"
#include <unordered_map>
#include <string>

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
    BackendRegistry::instance()
        .register_matmul(
            "CPU_NEON",
            arm_neon::matmul_neon
        );
}

static int neon_reg = (register_neon(),0);

}