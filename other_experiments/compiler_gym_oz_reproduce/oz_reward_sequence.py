import compiler_gym

from check_experiments.compiler_gym_oz_reproduce.actions import (
    OZ_FLAGS_SEQUENCE_NO_ANALYTICAL,
)
from utils import prepare_datasets

# OZ_FLAGS_SEQUENCE = [
#     "-ee-instrument",
#     "-simplifycfg",
#     "-sroa",
#     "-early-cse",
#     "-lower-expect",
#     "-forceattrs",
#     "-inferattrs",
#     "-ipsccp",
#     "-called-value-propagation",
#     "-attributor",
#     "-globalopt",
#     "-mem2reg",
#     "-deadargelim",
#     "-instcombine",
#     "-simplifycfg",
#     "-prune-eh",
#     "-inline",
#     "-functionattrs",
#     "-sroa",
#     "-early-cse-memssa",
#     "-speculative-execution",
#     "-jump-threading",
#     "-correlated-propagation",
#     "-simplifycfg",
#     "-instcombine",
#     "-tailcallelim",
#     "-simplifycfg",
#     "-reassociate",
#     "-loop-simplify",
#     "-lcssa",
#     "-loop-rotate",
#     "-licm",
#     "-loop-unswitch",
#     "-simplifycfg",
#     "-instcombine",
#     "-loop-simplify",
#     "-lcssa",
#     "-indvars",
#     "-loop-idiom",
#     "-loop-deletion",
#     "-loop-unroll",
#     "-mldst-motion",
#     "-gvn",
#     "-memcpyopt",
#     "-sccp",
#     "-bdce",
#     "-instcombine",
#     "-jump-threading",
#     "-correlated-propagation",
#     "-dse",
#     "-loop-simplify",
#     "-lcssa",
#     "-licm",
#     "-adce",
#     "-simplifycfg",
#     "-instcombine",
#     "-barrier",
#     "-elim-avail-extern",
#     "-rpo-functionattrs",
#     "-globalopt",
#     "-globaldce",
#     "-float2int",
#     "-lower-constant-intrinsics",
#     "-loop-simplify",
#     "-lcssa",
#     "-loop-rotate",
#     "-loop-distribute",
#     "-loop-vectorize",
#     "-loop-simplify",
#     "-loop-load-elim",
#     "-instcombine",
#     "-simplifycfg",
#     "-instcombine",
#     "-loop-simplify",
#     "-lcssa",
#     "-loop-unroll",
#     "-instcombine",
#     "-loop-simplify",
#     "-lcssa",
#     "-licm",
#     "-alignment-from-assumptions",
#     "-strip-dead-prototypes",
#     "-globaldce",
#     "-constmerge",
#     "-loop-simplify",
#     "-lcssa",
#     "-loop-sink",
#     "-instsimplify",
#     "-div-rem-pairs",
#     "-simplifycfg",
# ]

ENV = "llvm-v0"
REWARD_SPACE = "IrInstructionCountOz"
DATASETS = ["benchmark://cbench-v1"]


def main():
    with compiler_gym.make(ENV, reward_space=REWARD_SPACE) as env:
        train_benchmarks, _, _ = prepare_datasets(
            env,
            DATASETS,
            train_val_test_split=False,
        )
        for benchmark in train_benchmarks:
            env.reset(benchmark=benchmark)
            reward_sum = 0
            for flag in OZ_FLAGS_SEQUENCE_NO_ANALYTICAL:
                _, reward, _, _ = env.step(env.action_space.flags.index(flag))
                reward_sum += reward
            print(
                f'reward_sum: {reward_sum} - diff: {env.observation["IrInstructionCount"] / env.observation["IrInstructionCountOz"]} - {benchmark}'
            )


if __name__ == "__main__":
    main()

#

"""
Pass Arguments:  -tti -tbaa -scoped-noalias -assumption-cache-tracker -targetlibinfo -verify -ee-instrument -simplifycfg -domtree -sroa -early-cse -lower-expect
Pass Arguments:  -targetlibinfo -tti -tbaa -scoped-noalias -assumption-cache-tracker -profile-summary-info -forceattrs -inferattrs -ipsccp -called-value-propagation -attributor -globalopt -domtree -mem2reg -deadargelim -domtree -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -simplifycfg -basiccg -globals-aa -prune-eh -inline -functionattrs -domtree -sroa -basicaa -aa -memoryssa -early-cse-memssa -speculative-execution -basicaa -aa -lazy-value-info -jump-threading -correlated-propagation -simplifycfg -domtree -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -opt-remark-emitter -tailcallelim -simplifycfg -reassociate -domtree -loops -loop-simplify -lcssa-verification -lcssa -basicaa -aa -scalar-evolution -loop-rotate -memoryssa -licm -loop-unswitch -simplifycfg -domtree -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -loop-simplify -lcssa-verification -lcssa -scalar-evolution -indvars -loop-idiom -loop-deletion -loop-unroll -mldst-motion -phi-values -basicaa -aa -memdep -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -gvn -phi-values -basicaa -aa -memdep -memcpyopt -sccp -demanded-bits -bdce -basicaa -aa -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -lazy-value-info -jump-threading -correlated-propagation -basicaa -aa -phi-values -memdep -dse -basicaa -aa -memoryssa -loops -loop-simplify -lcssa-verification -lcssa -scalar-evolution -licm -postdomtree -adce -simplifycfg -domtree -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -barrier -elim-avail-extern -basiccg -rpo-functionattrs -globalopt -globaldce -basiccg -globals-aa -domtree -float2int -lower-constant-intrinsics -domtree -loops -loop-simplify -lcssa-verification -lcssa -basicaa -aa -scalar-evolution -loop-rotate -loop-accesses -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -loop-distribute -branch-prob -block-freq -scalar-evolution -basicaa -aa -loop-accesses -demanded-bits -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -loop-vectorize -loop-simplify -scalar-evolution -aa -loop-accesses -lazy-branch-prob -lazy-block-freq -loop-load-elim -basicaa -aa -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -simplifycfg -domtree -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -loop-simplify -lcssa-verification -lcssa -scalar-evolution -loop-unroll -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -memoryssa -loop-simplify -lcssa-verification -lcssa -scalar-evolution -licm -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -transform-warning -alignment-from-assumptions -strip-dead-prototypes -globaldce -constmerge -domtree -loops -branch-prob -block-freq -loop-simplify -lcssa-verification -lcssa -basicaa -aa -scalar-evolution -block-freq -loop-sink -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instsimplify -div-rem-pairs -simplifycfg -verify
Pass Arguments:  -domtree
Pass Arguments:  -targetlibinfo -domtree -loops -branch-prob -block-freq
Pass Arguments:  -targetlibinfo -domtree -loops -branch-prob -block-freq
"""
