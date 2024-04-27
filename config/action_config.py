POSET_RL_ODG = [
    "-instcombine -barrier -elim-avail-extern -rpo-functionattrs -globalopt -globaldce -constmerge",
    "-instcombine -barrier -elim-avail-extern -rpo-functionattrs -globalopt -globaldce -float2int -lower-constant-intrinsics",
    "-instcombine -barrier -elim-avail-extern -rpo-functionattrs -globalopt -mem2reg -deadargelim",
    "-instcombine -jump-threading -correlated-propagation -dse",
    "-instcombine -jump-threading -correlated-propagation",
    "-instcombine",
    "-instcombine -tailcallelim",
    "-loop-simplify -lcssa -indvars -loop-idiom -loop-deletion -loop-unroll",
    "-loop-simplify -lcssa -indvars -loop-idiom -loop-deletion -loop-unroll -mldst-motion -gvn -memcpyopt -sccp -bdce",
    "-loop-simplify -lcssa -licm -adce",
    "-loop-simplify -lcssa -licm -alignment-from-assumptions -strip-dead-prototypes -globaldce -constmerge",
    "-loop-simplify -lcssa -licm -alignment-from-assumptions -strip-dead-prototypes -globaldce -float2int -lower-constant-intrinsics",
    "-loop-simplify -lcssa -licm -loop-unswitch",
    "-loop-simplify -lcssa -loop-rotate -licm -adce",
    "-loop-simplify -lcssa -loop-rotate -licm -alignment-from-assumptions -strip-dead-prototypes -globaldce -constmerge",
    "-loop-simplify -lcssa -loop-rotate -licm -alignment-from-assumptions -strip-dead-prototypes -globaldce -float2int -lower-constant-intrinsics",
    "-loop-simplify -lcssa -loop-rotate -licm -loop-unswitch",
    "-loop-simplify -lcssa -loop-rotate -loop-distribute -loop-vectorize",
    "-loop-simplify -lcssa -loop-sink -instsimplify -div-rem-pairs -simplifycfg",
    "-loop-simplify -lcssa -loop-unroll",
    "-loop-simplify -lcssa -loop-unroll -mldst-motion -gvn -memcpyopt -sccp -bdce",
    "-loop-simplify -loop-load-elim",
    "-simplifycfg",
    "-simplifycfg -prune-eh -inline -functionattrs -sroa -early-cse -lower-expect -forceattrs -inferattrs -ipsccp -called-value-propagation -attributor -globalopt -globaldce -constmerge -barrier",
    "-simplifycfg -prune-eh -inline -functionattrs -sroa -early-cse -lower-expect -forceattrs -inferattrs -ipsccp -called-value-propagation -attributor -globalopt -globaldce -float2int -lower-constant-intrinsics -barrier",
    "-simplifycfg -prune-eh -inline -functionattrs -sroa -early-cse -lower-expect -forceattrs -inferattrs -ipsccp -called-value-propagation -attributor -globalopt -mem2reg -deadargelim -barrier",
    "-simplifycfg -prune-eh -inline -functionattrs -sroa -early-cse-memssa -speculative-execution -jump-threading -correlated-propagation -dse -barrier",
    "-simplifycfg -prune-eh -inline -functionattrs -sroa -early-cse-memssa -speculative-execution -jump-threading -correlated-propagation -barrier",
    "-simplifycfg -reassociate",
    "-simplifycfg -sroa -early-cse -lower-expect -forceattrs -inferattrs -ipsccp -called-value-propagation -attributor -globalopt -globaldce -constmerge",
    "-simplifycfg -sroa -early-cse -lower-expect -forceattrs -inferattrs -ipsccp -called-value-propagation -attributor -globalopt -globaldce -float2int -lower-constant-intrinsics",
    "-simplifycfg -sroa -early-cse -lower-expect -forceattrs -inferattrs -ipsccp -called-value-propagation -attributor -globalopt -mem2reg -deadargelim",
    "-simplifycfg -sroa -early-cse-memssa -speculative-execution -jump-threading -correlated-propagation -dse",
    "-simplifycfg -sroa -early-cse-memssa -speculative-execution -jump-threading -correlated-propagation",
]

POSET_RL_MANUAL = [
    "-ee-instrument -simplifycfg -sroa -early-cse -lower-expect -forceattrs -inferattrs -mem2reg",
    "-ipsccp -called-value-propagation -attributor -globalopt",
    "-deadargelim -instcombine -simplifycfg",
    "-prune-eh -inline -functionattrs -barrier",
    "-sroa -early-cse-memssa -speculative-execution -jump-threading -correlated-propagation",
    "-simplifycfg -instcombine -tailcallelim -simplifycfg -reassociate",
    "-loop-simplify -lcssa -loop-rotate -licm -loop-unswitch -simplifycfg -instcombine",
    "-loop-simplify -lcssa -indvars -loop-idiom -loop-deletion -loop-unroll",
    "-mldst-motion -gvn -memcpyopt -sccp -bdce -instcombine -jump-threading -correlated-propagation -dse",
    "-loop-simplify -lcssa -licm -adce -simplifycfg -instcombine",
    "-barrier -elim-avail-extern -rpo-functionattrs -globalopt -globaldce -float2int -lower-constant-intrinsics",
    "-loop-simplify -lcssa -loop-rotate -loop-distribute -loop-vectorize",
    "-loop-simplify -loop-load-elim -instcombine -simplifycfg -instcombine",
    "-loop-simplify -lcssa -loop-unroll -instcombine -loop-simplify -lcssa -licm -alignment-from-assumptions",
    "-strip-dead-prototypes -globaldce -constmerge -loop-simplify -lcssa -loop-sink -instsimplify -div-rem-pairs -simplifycfg",
]

O3_SUBSEQ_POSET_LIKE = [
    "-ee-instrument -simplifycfg -sroa -early-cse -lower-expect -forceattrs -inferattrs -callsite-splitting -ipsccp -called-value-propagation -attributor -globalopt -mem2reg",
    "-ipsccp -called-value-propagation -attributor -globalopt",
    "-deadargelim -instcombine -simplifycfg",
    "-prune-eh -inline -functionattrs -argpromotion",
    "-sroa -early-cse-memssa -speculative-execution -jump-threading -correlated-propagation",
    "-simplifycfg -aggressive-instcombine -instcombine -libcalls-shrinkwrap -pgo-memop-opt -tailcallelim -simplifycfg -reassociate",
    "-loop-simplify -lcssa -loop-rotate -licm -loop-unswitch -simplifycfg -instcombine",
    "-loop-simplify -lcssa -indvars -loop-idiom -loop-deletion -loop-unroll",
    "-mldst-motion -gvn -memcpyopt -sccp -bdce -instcombine -jump-threading -correlated-propagation -dse",
    "-loop-simplify -lcssa -licm -adce -simplifycfg -instcombine",
    "-barrier -elim-avail-extern -rpo-functionattrs -globalopt -globaldce -float2int -lower-constant-intrinsics",
    "-loop-simplify -lcssa -loop-rotate -loop-distribute -loop-vectorize",
    "-loop-simplify -loop-load-elim -instcombine",
    "-simplifycfg -slp-vectorizer -instcombine",
    "-loop-simplify -lcssa -loop-unroll -instcombine -loop-simplify -lcssa -licm -alignment-from-assumptions",
    "-strip-dead-prototypes -globaldce -constmerge -loop-simplify -lcssa -loop-sink -instsimplify -div-rem-pairs -simplifycfg",
]


O2_SUBSEQ_POSET_LIKE = [
    "-ee-instrument -simplifycfg -sroa -early-cse -lower-expect -forceattrs -inferattrs -ipsccp -called-value-propagation -attributor -globalopt -mem2reg",
    "-ipsccp -called-value-propagation -attributor -globalopt",
    "-deadargelim -instcombine -simplifycfg",
    "-prune-eh -inline -functionattrs",
    "-sroa -early-cse-memssa -speculative-execution -jump-threading -correlated-propagation",
    "-simplifycfg -instcombine -libcalls-shrinkwrap -pgo-memop-opt -tailcallelim -simplifycfg -reassociate",
    "-loop-simplify -lcssa -loop-rotate -licm -loop-unswitch -simplifycfg -instcombine",
    "-loop-simplify -lcssa -indvars -loop-idiom -loop-deletion -loop-unroll",
    "-mldst-motion -gvn -memcpyopt -sccp -bdce -instcombine -jump-threading -correlated-propagation -dse",
    "-loop-simplify -lcssa -licm -adce -simplifycfg -instcombine",
    "-barrier -elim-avail-extern -rpo-functionattrs -globalopt -globaldce -float2int -lower-constant-intrinsics",
    "-loop-simplify -lcssa -loop-rotate -loop-distribute -loop-vectorize",
    "-loop-simplify -loop-load-elim -instcombine",
    "-simplifycfg -slp-vectorizer -instcombine",
    "-loop-simplify -lcssa -loop-unroll -instcombine -loop-simplify -lcssa -licm -alignment-from-assumptions",
    "-strip-dead-prototypes -globaldce -constmerge -loop-simplify -lcssa -loop-sink -instsimplify -div-rem-pairs -simplifycfg",
]


O1_SUBSEQ_POSET_LIKE = [
    "-ee-instrument -simplifycfg -domtree -sroa -early-cse -lower-expect -targetlibinfo -tti -tbaa -scoped-noalias -assumption-cache-tracker -profile-summary-info -forceattrs -inferattrs -ipsccp -called-value-propagation -attributor -globalopt -domtree -mem2reg",
    "-ipsccp -called-value-propagation -attributor -globalopt",
    "-deadargelim -domtree -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -simplifycfg",
    "-prune-eh -always-inline -functionattrs",
    "-sroa -basicaa -aa -memoryssa -early-cse-memssa",
    "-simplifycfg -domtree -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -libcalls-shrinkwrap -loops -branch-prob -block-freq -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -pgo-memop-opt -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -tailcallelim -simplifycfg -reassociate",
    "-loop-simplify -lcssa-verification -lcssa -basicaa -aa -scalar-evolution -loop-rotate -memoryssa -licm -loop-unswitch -simplifycfg -domtree -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine",
    "-loop-simplify -lcssa-verification -lcssa -scalar-evolution -indvars -loop-idiom -loop-deletion -loop-unroll",
    "-phi-values -memdep -memcpyopt -sccp -demanded-bits -bdce -basicaa -aa -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine",
    "-postdomtree -adce -simplifycfg -domtree -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine",
    "-barrier -basiccg -rpo-functionattrs -globalopt -globaldce -basiccg -globals-aa -domtree -float2int -lower-constant-intrinsics",
    "-loop-simplify -lcssa-verification -lcssa -basicaa -aa -scalar-evolution -loop-rotate -loop-accesses -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -loop-distribute -branch-prob -block-freq -scalar-evolution -basicaa -aa -loop-accesses -demanded-bits -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -loop-vectorize",
    "-loop-simplify -scalar-evolution -aa -loop-accesses -lazy-branch-prob -lazy-block-freq -loop-load-elim -basicaa -aa -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine",
    "-simplifycfg -domtree -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine",
    "-loop-simplify -lcssa-verification -lcssa -scalar-evolution -loop-unroll -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -memoryssa -loop-simplify -lcssa-verification -lcssa -scalar-evolution -licm -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -transform-warning -alignment-from-assumptions",
    "-strip-dead-prototypes -domtree -loops -branch-prob -block-freq -loop-simplify -lcssa-verification -lcssa -basicaa -aa -scalar-evolution -block-freq -loop-sink -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instsimplify -div-rem-pairs -simplifycfg",
]

O23_SUBSEQ_POSET_LIKE = [
    "-ee-instrument -simplifycfg -sroa -early-cse -lower-expect -forceattrs -inferattrs -callsite-splitting -ipsccp -called-value-propagation -attributor -globalopt -mem2reg",
    "-ee-instrument -simplifycfg -sroa -early-cse -lower-expect -forceattrs -inferattrs -ipsccp -called-value-propagation -attributor -globalopt -mem2reg",
    "-ipsccp -called-value-propagation -attributor -globalopt",
    "-prune-eh -inline -functionattrs -argpromotion",
    "-prune-eh -inline -functionattrs",
    "-sroa -early-cse-memssa -speculative-execution -jump-threading -correlated-propagation",
    "-simplifycfg -aggressive-instcombine -instcombine -libcalls-shrinkwrap -pgo-memop-opt -tailcallelim -simplifycfg -reassociate",
    "-simplifycfg -instcombine -libcalls-shrinkwrap -pgo-memop-opt -tailcallelim -simplifycfg -reassociate",
    "-loop-simplify -lcssa -loop-rotate -licm -loop-unswitch -simplifycfg -instcombine",
    "-loop-simplify -lcssa -indvars -loop-idiom -loop-deletion -loop-unroll",
    "-mldst-motion -gvn -memcpyopt -sccp -bdce -instcombine -jump-threading -correlated-propagation -dse",
    "-loop-simplify -lcssa -licm -adce -simplifycfg -instcombine",
    "-barrier -elim-avail-extern -rpo-functionattrs -globalopt -globaldce -float2int -lower-constant-intrinsics",
    "-loop-simplify -lcssa -loop-rotate -loop-distribute -loop-vectorize",
    "-loop-simplify -loop-load-elim -instcombine",
    "-simplifycfg -slp-vectorizer -instcombine",
    "-loop-simplify -lcssa -loop-unroll -instcombine -loop-simplify -lcssa -licm -alignment-from-assumptions",
    "-strip-dead-prototypes -globaldce -constmerge -loop-simplify -lcssa -loop-sink -instsimplify -div-rem-pairs -simplifycfg",
]

# O23_SUBSEQ_POSET_LIKE = [
#     ['-ee-instrument', '-simplifycfg', '-sroa', '-early-cse', '-lower-expect', '-forceattrs', '-inferattrs',
#      '-callsite-splitting', '-ipsccp', '-called-value-propagation', '-attributor', '-globalopt', '-mem2reg'],
#     ['-ee-instrument', '-simplifycfg', '-sroa', '-early-cse', '-lower-expect', '-forceattrs', '-inferattrs', '-ipsccp',
#      '-called-value-propagation', '-attributor', '-globalopt', '-mem2reg'],
#     ['-ipsccp', '-called-value-propagation', '-attributor', '-globalopt'], ['-simplifycfg'],
#     ['-prune-eh', '-inline', '-functionattrs', '-argpromotion'], ['-prune-eh', '-inline', '-functionattrs'],
#     ['-sroa', '-early-cse-memssa', '-speculative-execution', '-jump-threading', '-correlated-propagation'],
#     ['-simplifycfg', '-aggressive-instcombine', '-instcombine', '-libcalls-shrinkwrap', '-pgo-memop-opt',
#      '-tailcallelim', '-simplifycfg', '-reassociate'],
#     ['-simplifycfg', '-instcombine', '-libcalls-shrinkwrap', '-pgo-memop-opt', '-tailcallelim', '-simplifycfg',
#      '-reassociate'],
#     ['-loop-simplify', '-lcssa', '-loop-rotate', '-licm', '-loop-unswitch', '-simplifycfg', '-instcombine'],
#     ['-loop-simplify', '-lcssa', '-indvars', '-loop-idiom', '-loop-deletion', '-loop-unroll'],
#     ['-mldst-motion', '-gvn', '-memcpyopt', '-sccp', '-bdce', '-instcombine', '-jump-threading',
#      '-correlated-propagation', '-dse'], ['-loop-simplify', '-lcssa', '-licm', '-adce', '-simplifycfg', '-instcombine'],
#     ['-barrier', '-elim-avail-extern', '-rpo-functionattrs', '-globalopt', '-globaldce', '-float2int',
#      '-lower-constant-intrinsics'], ['-loop-simplify', '-lcssa', '-loop-rotate', '-loop-distribute', '-loop-vectorize'],
#     ['-loop-simplify', '-loop-load-elim', '-instcombine'], ['-simplifycfg', '-slp-vectorizer', '-instcombine'],
#     ['-loop-simplify', '-lcssa', '-loop-unroll', '-instcombine', '-loop-simplify', '-lcssa', '-licm',
#      '-alignment-from-assumptions'],
#     ['-strip-dead-prototypes', '-globaldce', '-constmerge', '-loop-simplify', '-lcssa', '-loop-sink', '-instsimplify',
#      '-div-rem-pairs', '-simplifycfg']
# ]

# https://github.com/facebookresearch/CompilerGym/blob/development/leaderboard/llvm_instcount/dqn/README.md
COMPILER_GYM_LEADERBOARD_DQN_ACTION_SET = [
    "-break-crit-edges",
    "-early-cse-memssa",
    "-gvn-hoist",
    "-gvn",
    "-instcombine",
    "-instsimplify",
    "-jump-threading",
    "-loop-reduce",
    "-loop-rotate",
    "-loop-versioning",
    "-mem2reg",
    "-newgvn",
    "-reg2mem",
    "-simplifycfg",
    "-sroa",
]

COMPLETE_ACTION_SET = [
    "-add-discriminators",
    "-adce",
    "-aggressive-instcombine",
    "-alignment-from-assumptions",
    "-always-inline",
    "-argpromotion",
    "-attributor",
    "-barrier",
    "-bdce",
    "-break-crit-edges",
    "-simplifycfg",
    "-callsite-splitting",
    "-called-value-propagation",
    "-canonicalize-aliases",
    "-consthoist",
    "-constmerge",
    "-constprop",
    "-coro-cleanup",
    "-coro-early",
    "-coro-elide",
    "-coro-split",
    "-correlated-propagation",
    "-cross-dso-cfi",
    "-deadargelim",
    "-dce",
    "-die",
    "-dse",
    "-reg2mem",
    "-div-rem-pairs",
    "-early-cse-memssa",
    "-early-cse",
    "-elim-avail-extern",
    "-ee-instrument",
    "-flattencfg",
    "-float2int",
    "-forceattrs",
    "-inline",
    "-insert-gcov-profiling",
    "-gvn-hoist",
    "-gvn",
    "-globaldce",
    "-globalopt",
    "-globalsplit",
    "-guard-widening",
    "-hotcoldsplit",
    "-ipconstprop",
    "-ipsccp",
    "-indvars",
    "-irce",
    "-infer-address-spaces",
    "-inferattrs",
    "-inject-tli-mappings",
    "-instsimplify",
    "-instcombine",
    "-instnamer",
    "-jump-threading",
    "-lcssa",
    "-licm",
    "-libcalls-shrinkwrap",
    "-load-store-vectorizer",
    "-loop-data-prefetch",
    "-loop-deletion",
    "-loop-distribute",
    "-loop-fusion",
    "-loop-guard-widening",
    "-loop-idiom",
    "-loop-instsimplify",
    "-loop-interchange",
    "-loop-load-elim",
    "-loop-predication",
    "-loop-reroll",
    "-loop-rotate",
    "-loop-simplifycfg",
    "-loop-simplify",
    "-loop-sink",
    "-loop-reduce",
    "-loop-unroll-and-jam",
    "-loop-unroll",
    "-loop-unswitch",
    "-loop-vectorize",
    "-loop-versioning-licm",
    "-loop-versioning",
    "-loweratomic",
    "-lower-constant-intrinsics",
    "-lower-expect",
    "-lower-guard-intrinsic",
    "-lowerinvoke",
    "-lower-matrix-intrinsics",
    "-lowerswitch",
    "-lower-widenable-condition",
    "-memcpyopt",
    "-mergefunc",
    "-mergeicmps",
    "-mldst-motion",
    "-sancov",
    "-name-anon-globals",
    "-nary-reassociate",
    "-newgvn",
    "-pgo-memop-opt",
    "-partial-inliner",
    "-partially-inline-libcalls",
    "-post-inline-ee-instrument",
    "-functionattrs",
    "-mem2reg",
    "-prune-eh",
    "-reassociate",
    "-redundant-dbg-inst-elim",
    "-rpo-functionattrs",
    "-rewrite-statepoints-for-gc",
    "-sccp",
    "-slp-vectorizer",
    "-sroa",
    "-scalarizer",
    "-separate-const-offset-from-gep",
    "-simple-loop-unswitch",
    "-sink",
    "-speculative-execution",
    "-slsr",
    "-strip-dead-prototypes",
    "-strip-debug-declare",
    "-strip-nondebug",
    "-strip",
    "-tailcallelim",
    "-mergereturn",
]

COMPLETE_ACTION_SET_WITH_ANALYTICAL = []

O3_ACTION_SET = [
    "-forceattrs",
    "-loop-rotate",
    "-loop-simplify",
    "-loop-unswitch",
    "-inferattrs",
    "-prune-eh",
    "-callsite-splitting",
    "-elim-avail-extern",
    "-float2int",
    "-loop-load-elim",
    "-licm",
    "-mldst-motion",
    "-tailcallelim",
    "-loop-idiom",
    "-functionattrs",
    "-sccp",
    "-loop-vectorize",
    "-dse",
    "-mem2reg",
    "-argpromotion",
    "-rpo-functionattrs",
    "-alignment-from-assumptions",
    "-ipsccp",
    "-speculative-execution",
    "-loop-unroll",
    "-loop-deletion",
    "-lower-expect",
    "-bdce",
    "-early-cse-memssa",
    "-constmerge",
    "-deadargelim",
    "-called-value-propagation",
    "-indvars",
    "-instsimplify",
    "-lcssa",
    "-lower-constant-intrinsics",
    "-correlated-propagation",
    "-inline",
    "-ee-instrument",
    "-aggressive-instcombine",
    "-attributor",
    "-simplifycfg",
    "-loop-sink",
    "-reassociate",
    "-globalopt",
    "-instcombine",
    "-loop-distribute",
    "-slp-vectorizer",
    "-gvn",
    "-adce",
    "-pgo-memop-opt",
    "-sroa",
    "-jump-threading",
    "-memcpyopt",
    "-strip-dead-prototypes",
    "-div-rem-pairs",
    "-barrier",
    "-globaldce",
    "-early-cse",
    "-libcalls-shrinkwrap",
]

O3_SEQ = [
    "-tti",
    "-tbaa",
    "-scoped-noalias",
    "-assumption-cache-tracker",
    "-targetlibinfo",
    "-verify",
    "-ee-instrument",
    "-simplifycfg",
    "-domtree",
    "-sroa",
    "-early-cse",
    "-lower-expect",
    "-targetlibinfo",
    "-tti",
    "-tbaa",
    "-scoped-noalias",
    "-assumption-cache-tracker",
    "-profile-summary-info",
    "-forceattrs",
    "-inferattrs",
    "-domtree",
    "-callsite-splitting",
    "-ipsccp",
    "-called-value-propagation",
    "-attributor",
    "-globalopt",
    "-domtree",
    "-mem2reg",
    "-deadargelim",
    "-domtree",
    "-basicaa",
    "-aa",
    "-loops",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-simplifycfg",
    "-basiccg",
    "-globals-aa",
    "-prune-eh",
    "-inline",
    "-functionattrs",
    "-argpromotion",
    "-domtree",
    "-sroa",
    "-basicaa",
    "-aa",
    "-memoryssa",
    "-early-cse-memssa",
    "-speculative-execution",
    "-basicaa",
    "-aa",
    "-lazy-value-info",
    "-jump-threading",
    "-correlated-propagation",
    "-simplifycfg",
    "-domtree",
    "-aggressive-instcombine",
    "-basicaa",
    "-aa",
    "-loops",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-libcalls-shrinkwrap",
    "-loops",
    "-branch-prob",
    "-block-freq",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-pgo-memop-opt",
    "-basicaa",
    "-aa",
    "-loops",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-tailcallelim",
    "-simplifycfg",
    "-reassociate",
    "-domtree",
    "-loops",
    "-loop-simplify",
    "-lcssa-verification",
    "-lcssa",
    "-basicaa",
    "-aa",
    "-scalar-evolution",
    "-loop-rotate",
    "-memoryssa",
    "-licm",
    "-loop-unswitch",
    "-simplifycfg",
    "-domtree",
    "-basicaa",
    "-aa",
    "-loops",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-loop-simplify",
    "-lcssa-verification",
    "-lcssa",
    "-scalar-evolution",
    "-indvars",
    "-loop-idiom",
    "-loop-deletion",
    "-loop-unroll",
    "-mldst-motion",
    "-phi-values",
    "-basicaa",
    "-aa",
    "-memdep",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-gvn",
    "-phi-values",
    "-basicaa",
    "-aa",
    "-memdep",
    "-memcpyopt",
    "-sccp",
    "-demanded-bits",
    "-bdce",
    "-basicaa",
    "-aa",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-lazy-value-info",
    "-jump-threading",
    "-correlated-propagation",
    "-basicaa",
    "-aa",
    "-phi-values",
    "-memdep",
    "-dse",
    "-basicaa",
    "-aa",
    "-memoryssa",
    "-loops",
    "-loop-simplify",
    "-lcssa-verification",
    "-lcssa",
    "-scalar-evolution",
    "-licm",
    "-postdomtree",
    "-adce",
    "-simplifycfg",
    "-domtree",
    "-basicaa",
    "-aa",
    "-loops",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-barrier",
    "-elim-avail-extern",
    "-basiccg",
    "-rpo-functionattrs",
    "-globalopt",
    "-globaldce",
    "-basiccg",
    "-globals-aa",
    "-domtree",
    "-float2int",
    "-lower-constant-intrinsics",
    "-domtree",
    "-loops",
    "-loop-simplify",
    "-lcssa-verification",
    "-lcssa",
    "-basicaa",
    "-aa",
    "-scalar-evolution",
    "-loop-rotate",
    "-loop-accesses",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-loop-distribute",
    "-branch-prob",
    "-block-freq",
    "-scalar-evolution",
    "-basicaa",
    "-aa",
    "-loop-accesses",
    "-demanded-bits",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-loop-vectorize",
    "-loop-simplify",
    "-scalar-evolution",
    "-aa",
    "-loop-accesses",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-loop-load-elim",
    "-basicaa",
    "-aa",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-simplifycfg",
    "-domtree",
    "-loops",
    "-scalar-evolution",
    "-basicaa",
    "-aa",
    "-demanded-bits",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-slp-vectorizer",
    "-opt-remark-emitter",
    "-instcombine",
    "-loop-simplify",
    "-lcssa-verification",
    "-lcssa",
    "-scalar-evolution",
    "-loop-unroll",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-memoryssa",
    "-loop-simplify",
    "-lcssa-verification",
    "-lcssa",
    "-scalar-evolution",
    "-licm",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-transform-warning",
    "-alignment-from-assumptions",
    "-strip-dead-prototypes",
    "-globaldce",
    "-constmerge",
    "-domtree",
    "-loops",
    "-branch-prob",
    "-block-freq",
    "-loop-simplify",
    "-lcssa-verification",
    "-lcssa",
    "-basicaa",
    "-aa",
    "-scalar-evolution",
    "-block-freq",
    "-loop-sink",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instsimplify",
    "-div-rem-pairs",
    "-simplifycfg",
    "-verify",
    "-domtree",
    "-targetlibinfo",
    "-domtree",
    "-loops",
    "-branch-prob",
    "-block-freq",
    "-targetlibinfo",
    "-domtree",
    "-loops",
    "-branch-prob",
    "-block-freq",
]

O2_SEQ = [
    "-tti",
    "-tbaa",
    "-scoped-noalias-aa",
    "-assumption-cache-tracker",
    "-targetlibinfo",
    "-verify",
    "-lower-expect",
    "-simplifycfg",
    "-domtree",
    "-sroa",
    "-early-cse",
    "-targetlibinfo",
    "-tti",
    "-tbaa",
    "-scoped-noalias-aa",
    "-assumption-cache-tracker",
    "-profile-summary-info",
    "-annotation2metadata",
    "-forceattrs",
    "-inferattrs",
    "-ipsccp",
    "-called-value-propagation",
    "-globalopt",
    "-domtree",
    "-mem2reg",
    "-deadargelim",
    "-domtree",
    "-basic-aa",
    "-aa",
    "-loops",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-simplifycfg",
    "-basiccg",
    "-globals-aa",
    "-prune-eh",
    "-inline",
    "-openmp-opt-cgscc",
    "-function-attrs",
    "-domtree",
    "-sroa",
    "-basic-aa",
    "-aa",
    "-memoryssa",
    "-early-cse-memssa",
    "-speculative-execution",
    "-aa",
    "-lazy-value-info",
    "-jump-threading",
    "-correlated-propagation",
    "-simplifycfg",
    "-domtree",
    "-basic-aa",
    "-aa",
    "-loops",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-libcalls-shrinkwrap",
    "-loops",
    "-postdomtree",
    "-branch-prob",
    "-block-freq",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-pgo-memop-opt",
    "-basic-aa",
    "-aa",
    "-loops",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-tailcallelim",
    "-simplifycfg",
    "-reassociate",
    "-domtree",
    "-basic-aa",
    "-aa",
    "-memoryssa",
    "-loops",
    "-loop-simplify",
    "-lcssa-verification",
    "-lcssa",
    "-scalar-evolution",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-licm",
    "-loop-rotate",
    "-licm",
    "-loop-unswitch",
    "-simplifycfg",
    "-domtree",
    "-basic-aa",
    "-aa",
    "-loops",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-loop-simplify",
    "-lcssa-verification",
    "-lcssa",
    "-scalar-evolution",
    "-loop-idiom",
    "-indvars",
    "-loop-deletion",
    "-loop-unroll",
    "-sroa",
    "-aa",
    "-mldst-motion",
    "-phi-values",
    "-aa",
    "-memdep",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-gvn",
    "-sccp",
    "-demanded-bits",
    "-bdce",
    "-basic-aa",
    "-aa",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-lazy-value-info",
    "-jump-threading",
    "-correlated-propagation",
    "-postdomtree",
    "-adce",
    "-basic-aa",
    "-aa",
    "-memoryssa",
    "-memcpyopt",
    "-loops",
    "-dse",
    "-loop-simplify",
    "-lcssa-verification",
    "-lcssa",
    "-aa",
    "-scalar-evolution",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-licm",
    "-simplifycfg",
    "-domtree",
    "-basic-aa",
    "-aa",
    "-loops",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-barrier",
    "-elim-avail-extern",
    "-basiccg",
    "-rpo-function-attrs",
    "-globalopt",
    "-globaldce",
    "-basiccg",
    "-globals-aa",
    "-domtree",
    "-float2int",
    "-lower-constant-intrinsics",
    "-loops",
    "-loop-simplify",
    "-lcssa-verification",
    "-lcssa",
    "-basic-aa",
    "-aa",
    "-scalar-evolution",
    "-loop-rotate",
    "-loop-accesses",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-loop-distribute",
    "-postdomtree",
    "-branch-prob",
    "-block-freq",
    "-scalar-evolution",
    "-basic-aa",
    "-aa",
    "-loop-accesses",
    "-demanded-bits",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-inject-tli-mappings",
    "-loop-vectorize",
    "-loop-simplify",
    "-scalar-evolution",
    "-aa",
    "-loop-accesses",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-loop-load-elim",
    "-basic-aa",
    "-aa",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-simplifycfg",
    "-domtree",
    "-loops",
    "-scalar-evolution",
    "-basic-aa",
    "-aa",
    "-demanded-bits",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-inject-tli-mappings",
    "-slp-vectorizer",
    "-vector-combine",
    "-opt-remark-emitter",
    "-instcombine",
    "-loop-simplify",
    "-lcssa-verification",
    "-lcssa",
    "-scalar-evolution",
    "-loop-unroll",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-memoryssa",
    "-loop-simplify",
    "-lcssa-verification",
    "-lcssa",
    "-scalar-evolution",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-licm",
    "-opt-remark-emitter",
    "-transform-warning",
    "-alignment-from-assumptions",
    "-strip-dead-prototypes",
    "-globaldce",
    "-constmerge",
    "-cg-profile",
    "-domtree",
    "-loops",
    "-postdomtree",
    "-branch-prob",
    "-block-freq",
    "-loop-simplify",
    "-lcssa-verification",
    "-lcssa",
    "-basic-aa",
    "-aa",
    "-scalar-evolution",
    "-block-freq",
    "-loop-sink",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instsimplify",
    "-div-rem-pairs",
    "-simplifycfg",
    "-annotation-remarks",
    "-verify",
    "-domtree",
    "-targetlibinfo",
    "-domtree",
    "-loops",
    "-postdomtree",
    "-branch-prob",
    "-block-freq",
    "-targetlibinfo",
    "-domtree",
    "-loops",
    "-postdomtree",
    "-branch-prob",
    "-block-freq",
    "-targetlibinfo",
    "-domtree",
    "-loops",
    "-lazy-branch-prob",
    "-lazy-block-freq",
]

O1_SEQ = [
    "-tti",
    "-tbaa",
    "-scoped-noalias-aa",
    "-assumption-cache-tracker",
    "-targetlibinfo",
    "-verify",
    "-lower-expect",
    "-simplifycfg",
    "-domtree",
    "-sroa",
    "-early-cse",
    "-targetlibinfo",
    "-tti",
    "-tbaa",
    "-scoped-noalias-aa",
    "-assumption-cache-tracker",
    "-profile-summary-info",
    "-annotation2metadata",
    "-forceattrs",
    "-inferattrs",
    "-ipsccp",
    "-called-value-propagation",
    "-globalopt",
    "-domtree",
    "-mem2reg",
    "-deadargelim",
    "-domtree",
    "-basic-aa",
    "-aa",
    "-loops",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-simplifycfg",
    "-basiccg",
    "-globals-aa",
    "-prune-eh",
    "-always-inline",
    "-function-attrs",
    "-domtree",
    "-sroa",
    "-basic-aa",
    "-aa",
    "-memoryssa",
    "-early-cse-memssa",
    "-simplifycfg",
    "-domtree",
    "-basic-aa",
    "-aa",
    "-loops",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-libcalls-shrinkwrap",
    "-loops",
    "-postdomtree",
    "-branch-prob",
    "-block-freq",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-pgo-memop-opt",
    "-simplifycfg",
    "-reassociate",
    "-domtree",
    "-basic-aa",
    "-aa",
    "-memoryssa",
    "-loops",
    "-loop-simplify",
    "-lcssa-verification",
    "-lcssa",
    "-scalar-evolution",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-licm",
    "-loop-rotate",
    "-licm",
    "-loop-unswitch",
    "-simplifycfg",
    "-domtree",
    "-basic-aa",
    "-aa",
    "-loops",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-loop-simplify",
    "-lcssa-verification",
    "-lcssa",
    "-scalar-evolution",
    "-loop-idiom",
    "-indvars",
    "-loop-deletion",
    "-loop-unroll",
    "-sroa",
    "-sccp",
    "-demanded-bits",
    "-bdce",
    "-aa",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-postdomtree",
    "-adce",
    "-basic-aa",
    "-aa",
    "-memoryssa",
    "-memcpyopt",
    "-simplifycfg",
    "-domtree",
    "-basic-aa",
    "-aa",
    "-loops",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-barrier",
    "-basiccg",
    "-rpo-function-attrs",
    "-globalopt",
    "-globaldce",
    "-basiccg",
    "-globals-aa",
    "-domtree",
    "-float2int",
    "-lower-constant-intrinsics",
    "-loops",
    "-loop-simplify",
    "-lcssa-verification",
    "-lcssa",
    "-basic-aa",
    "-aa",
    "-scalar-evolution",
    "-loop-rotate",
    "-loop-accesses",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-loop-distribute",
    "-postdomtree",
    "-branch-prob",
    "-block-freq",
    "-scalar-evolution",
    "-basic-aa",
    "-aa",
    "-loop-accesses",
    "-demanded-bits",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-inject-tli-mappings",
    "-loop-vectorize",
    "-loop-simplify",
    "-scalar-evolution",
    "-aa",
    "-loop-accesses",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-loop-load-elim",
    "-basic-aa",
    "-aa",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-simplifycfg",
    "-domtree",
    "-basic-aa",
    "-aa",
    "-vector-combine",
    "-loops",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-loop-simplify",
    "-lcssa-verification",
    "-lcssa",
    "-scalar-evolution",
    "-loop-unroll",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instcombine",
    "-memoryssa",
    "-loop-simplify",
    "-lcssa-verification",
    "-lcssa",
    "-scalar-evolution",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-licm",
    "-opt-remark-emitter",
    "-transform-warning",
    "-alignment-from-assumptions",
    "-strip-dead-prototypes",
    "-cg-profile",
    "-domtree",
    "-loops",
    "-postdomtree",
    "-branch-prob",
    "-block-freq",
    "-loop-simplify",
    "-lcssa-verification",
    "-lcssa",
    "-basic-aa",
    "-aa",
    "-scalar-evolution",
    "-block-freq",
    "-loop-sink",
    "-lazy-branch-prob",
    "-lazy-block-freq",
    "-opt-remark-emitter",
    "-instsimplify",
    "-div-rem-pairs",
    "-simplifycfg",
    "-annotation-remarks",
    "-verify",
    "-domtree",
    "-targetlibinfo",
    "-domtree",
    "-loops",
    "-postdomtree",
    "-branch-prob",
    "-block-freq",
    "-targetlibinfo",
    "-domtree",
    "-loops",
    "-postdomtree",
    "-branch-prob",
    "-block-freq",
    "-targetlibinfo",
    "-domtree",
    "-loops",
    "-lazy-branch-prob",
    "-lazy-block-freq",
]
