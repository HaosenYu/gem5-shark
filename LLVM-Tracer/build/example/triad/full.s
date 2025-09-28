	.text
	.file	"llvm-link"
	.globl	triad                   # -- Begin function triad
	.p2align	4, 0x90
	.type	triad,@function
triad:                                  # @triad
.Lfunc_begin0:
	.file	1 "/workspace/LLVM-Tracer/example/triad/triad.c"
	.loc	1 3 0                   # /workspace/LLVM-Tracer/example/triad/triad.c:3:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r12
	pushq	%rbx
	subq	$304, %rsp              # imm = 0x130
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	movabsq	$.L.str.4, %rax
	xorl	%r8d, %r8d
	movl	%r8d, %r9d
	movl	$1, %r8d
.Ltmp0:
	.loc	1 3 17 prologue_end     # /workspace/LLVM-Tracer/example/triad/triad.c:3:17
	movq	%rdi, -40(%rbp)         # 8-byte Spill
	movq	%rax, %rdi
	movq	%rsi, -48(%rbp)         # 8-byte Spill
	movq	%r9, %rsi
	movq	%rdx, -56(%rbp)         # 8-byte Spill
	movl	%r8d, %edx
	movl	%ecx, -60(%rbp)         # 4-byte Spill
	movl	%r8d, %ecx
	callq	trace_logger_update_status
	movabsq	$.L.str.4, %rdi
	movl	$4, %ecx
	movl	%ecx, %esi
	callq	trace_logger_log_entry
	movl	$1, %ecx
	movl	%ecx, %eax
	movl	$64, %ecx
	movl	%ecx, %esi
	movabsq	$.L.str.13, %r8
	xorl	%ecx, %ecx
	movl	%ecx, %r9d
	movabsq	$.L.str.8, %rdi
	movq	%rdi, -72(%rbp)         # 8-byte Spill
	movq	%rax, %rdi
	movq	-40(%rbp), %rdx         # 8-byte Reload
	movq	%rax, %rcx
	movq	-72(%rbp), %rax         # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$2, %r10d
	movl	%r10d, %edi
	movl	$64, %r10d
	movl	%r10d, %esi
	movl	$1, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.16, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rax
	movq	-48(%rbp), %rdx         # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$3, %r10d
	movl	%r10d, %edi
	movl	$64, %r10d
	movl	%r10d, %esi
	movl	$1, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.22, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rax
	movq	-56(%rbp), %rdx         # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$4, %r10d
	movl	%r10d, %edi
	movl	$32, %r10d
	movl	%r10d, %esi
	movl	$1, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.19, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rax
	movl	-60(%rbp), %r10d        # 4-byte Reload
	movl	%r10d, %r11d
	movl	%r11d, %edx
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$5, %r10d
	movl	%r10d, %edi
	movabsq	$.L.str.4, %rsi
	movabsq	$.L.str.5, %rdx
	movabsq	$.L.str.6, %rcx
	movl	$2, %r10d
	movl	%r10d, %r8d
	movl	$1, %r9d
	#DEBUG_VALUE: triad:a <- [DW_OP_constu 40, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: triad:b <- [DW_OP_constu 48, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: triad:c <- [DW_OP_constu 56, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: triad:s <- [DW_OP_constu 60, DW_OP_minus] [%rbp+0]
.Ltmp1:
	#DEBUG_VALUE: triad:i <- 0
	.loc	1 5 7                   # /workspace/LLVM-Tracer/example/triad/triad.c:5:7
	movl	$1, (%rsp)
	callq	trace_logger_log0
	movl	$1, %r9d
	movl	%r9d, %eax
	xorl	%r9d, %r9d
	movl	%r9d, %ecx
	movabsq	$.L.str.7, %r8
	movabsq	$.L.str.8, %rdx
	movq	%rax, %rdi
	movq	%rcx, %rsi
	movq	%rdx, -80(%rbp)         # 8-byte Spill
	movq	%rcx, %rdx
	movq	%rcx, -88(%rbp)         # 8-byte Spill
	movq	%rax, %rcx
	movq	-88(%rbp), %r9          # 8-byte Reload
	movq	-80(%rbp), %rax         # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	xorl	%r10d, %r10d
	movl	%r10d, %eax
	movq	%rax, -96(%rbp)         # 8-byte Spill
.Ltmp2:
.LBB0_1:                                # =>This Inner Loop Header: Depth=1
	#DEBUG_VALUE: triad:s <- [DW_OP_constu 60, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: triad:c <- [DW_OP_constu 56, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: triad:b <- [DW_OP_constu 48, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: triad:a <- [DW_OP_constu 40, DW_OP_minus] [%rbp+0]
	.loc	1 0 7 is_stmt 0         # /workspace/LLVM-Tracer/example/triad/triad.c:0:7
	movq	-96(%rbp), %rax         # 8-byte Reload
	movq	$-1, %rdi
	movabsq	$.L.str.4, %rsi
	movabsq	$.L.str.7, %rdx
	movabsq	$.L.str.9, %rcx
	movl	$53, %r8d
                                        # kill: def %r8 killed %r8d
	movl	$1, %r9d
	.loc	1 4 6 is_stmt 1         # /workspace/LLVM-Tracer/example/triad/triad.c:4:6
	movl	$1, (%rsp)
	movq	%rax, -104(%rbp)        # 8-byte Spill
	callq	trace_logger_log0
	movl	$2, %r9d
	movl	%r9d, %edi
	movl	$64, %r9d
	movl	%r9d, %esi
	xorl	%r9d, %r9d
	movl	%r9d, %edx
	movl	$1, %r9d
	movl	%r9d, %eax
	movabsq	$.L.str.10, %r8
	movabsq	$.L.str.7, %rcx
	movq	%rcx, -112(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	%rax, %r9
	movq	-112(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %r10d
	movl	%r10d, %eax
	movl	$64, %r10d
	movl	%r10d, %esi
	xorl	%r10d, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.11, %r8
	movabsq	$.L.str.5, %rdx
	movq	%rax, %rdi
	movq	%rdx, -120(%rbp)        # 8-byte Spill
	movq	%rcx, %rdx
	movq	%rax, %r9
	movq	-120(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$19134, %r10d           # imm = 0x4ABE
	movl	%r10d, %edi
	movl	$64, %r10d
	movl	%r10d, %esi
	movl	$1, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.9, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.5, %rax
	movq	-104(%rbp), %rdx        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$6, %r10d
	movl	%r10d, %edi
	movabsq	$.L.str.4, %rsi
	movabsq	$.L.str.7, %rdx
	movabsq	$.L.str.12, %rcx
	movl	$32, %r10d
	movl	%r10d, %r8d
	movl	$1, %r9d
.Ltmp3:
	#DEBUG_VALUE: triad:i <- [DW_OP_constu 104, DW_OP_minus] [%rbp+0]
	.loc	1 6 10                  # /workspace/LLVM-Tracer/example/triad/triad.c:6:10
	movl	$1, (%rsp)
	callq	trace_logger_log0
	movl	$2, %r9d
	movl	%r9d, %edi
	movl	$64, %r9d
	movl	%r9d, %esi
	movl	$1, %r9d
	movl	%r9d, %ecx
	movabsq	$.L.str.9, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rax
	movq	-104(%rbp), %rdx        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %r10d
	movl	%r10d, %eax
	movl	$64, %r10d
	movl	%r10d, %esi
	movabsq	$.L.str.13, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rcx
	movq	%rax, %rdi
	movq	-40(%rbp), %rdx         # 8-byte Reload
	movq	%rcx, -128(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-128(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$19134, %r10d           # imm = 0x4ABE
	movl	%r10d, %edi
	movl	$64, %r10d
	movl	%r10d, %esi
	movl	$1, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.12, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rax
	movq	-104(%rbp), %rdx        # 8-byte Reload
	shlq	$2, %rdx
	movq	-40(%rbp), %r11         # 8-byte Reload
	addq	%rdx, %r11
	movq	%r11, %rdx
	movq	%rax, (%rsp)
	movq	%r11, -136(%rbp)        # 8-byte Spill
	callq	trace_logger_log_ptr
	movl	$6, %r10d
	movl	%r10d, %edi
	movabsq	$.L.str.4, %rsi
	movabsq	$.L.str.7, %rdx
	movabsq	$.L.str.14, %rcx
	movl	$30, %r10d
	movl	%r10d, %r8d
	movl	$1, %r9d
	movl	$1, (%rsp)
	callq	trace_logger_log0
	movl	$1, %r9d
	movl	%r9d, %eax
	movl	$64, %r9d
	movl	%r9d, %esi
	movabsq	$.L.str.12, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rcx
	movq	%rax, %rdi
	movq	-136(%rbp), %rdx        # 8-byte Reload
	movq	%rcx, -144(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-144(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$19134, %r10d           # imm = 0x4ABE
	movl	%r10d, %edi
	movl	$32, %r10d
	movl	%r10d, %esi
	movl	$1, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.14, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rax
	movq	-40(%rbp), %rdx         # 8-byte Reload
	movq	-104(%rbp), %r11        # 8-byte Reload
	movl	(%rdx,%r11,4), %r10d
	.loc	1 6 19 is_stmt 0        # /workspace/LLVM-Tracer/example/triad/triad.c:6:19
	movl	%r10d, %ebx
	movl	%ebx, %edx
	movq	%rax, (%rsp)
	movl	%r10d, -148(%rbp)       # 4-byte Spill
	callq	trace_logger_log_int
	movl	$6, %r10d
	movl	%r10d, %edi
	movabsq	$.L.str.4, %rsi
	movabsq	$.L.str.7, %rdx
	movabsq	$.L.str.15, %rcx
	movl	$32, %r10d
	movl	%r10d, %r8d
	movl	$1, %r9d
	movl	$1, (%rsp)
	callq	trace_logger_log0
	movl	$2, %r9d
	movl	%r9d, %edi
	movl	$64, %r9d
	movl	%r9d, %esi
	movl	$1, %r9d
	movl	%r9d, %ecx
	movabsq	$.L.str.9, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rax
	movq	-104(%rbp), %rdx        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %r10d
	movl	%r10d, %eax
	movl	$64, %r10d
	movl	%r10d, %esi
	movabsq	$.L.str.16, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rcx
	movq	%rax, %rdi
	movq	-48(%rbp), %rdx         # 8-byte Reload
	movq	%rcx, -160(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-160(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$19134, %r10d           # imm = 0x4ABE
	movl	%r10d, %edi
	movl	$64, %r10d
	movl	%r10d, %esi
	movl	$1, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.15, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rax
	movq	-104(%rbp), %rdx        # 8-byte Reload
	shlq	$2, %rdx
	movq	-48(%rbp), %r11         # 8-byte Reload
	addq	%rdx, %r11
	movq	%r11, %rdx
	movq	%rax, (%rsp)
	movq	%r11, -168(%rbp)        # 8-byte Spill
	callq	trace_logger_log_ptr
	movl	$6, %r10d
	movl	%r10d, %edi
	movabsq	$.L.str.4, %rsi
	movabsq	$.L.str.7, %rdx
	movabsq	$.L.str.17, %rcx
	movl	$30, %r10d
	movl	%r10d, %r8d
	movl	$1, %r9d
	movl	$1, (%rsp)
	callq	trace_logger_log0
	movl	$1, %r9d
	movl	%r9d, %eax
	movl	$64, %r9d
	movl	%r9d, %esi
	movabsq	$.L.str.15, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rcx
	movq	%rax, %rdi
	movq	-168(%rbp), %rdx        # 8-byte Reload
	movq	%rcx, -176(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-176(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$19134, %r10d           # imm = 0x4ABE
	movl	%r10d, %edi
	movl	$32, %r10d
	movl	%r10d, %esi
	movl	$1, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.17, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rax
	movq	-48(%rbp), %rdx         # 8-byte Reload
	movq	-104(%rbp), %r11        # 8-byte Reload
	movl	(%rdx,%r11,4), %r10d
	.loc	1 6 18                  # /workspace/LLVM-Tracer/example/triad/triad.c:6:18
	movl	%r10d, %ebx
	movl	%ebx, %edx
	movq	%rax, (%rsp)
	movl	%r10d, -180(%rbp)       # 4-byte Spill
	callq	trace_logger_log_int
	movl	$6, %r10d
	movl	%r10d, %edi
	movabsq	$.L.str.4, %rsi
	movabsq	$.L.str.7, %rdx
	movabsq	$.L.str.18, %rcx
	movl	$15, %r10d
	movl	%r10d, %r8d
	movl	$1, %r9d
	movl	$1, (%rsp)
	callq	trace_logger_log0
	movl	$2, %r9d
	movl	%r9d, %edi
	movl	$32, %r9d
	movl	%r9d, %esi
	movl	$1, %r9d
	movl	%r9d, %ecx
	movabsq	$.L.str.19, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rax
	movl	-60(%rbp), %r10d        # 4-byte Reload
	movl	%r10d, %ebx
	movl	%ebx, %edx
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %r10d
	movl	%r10d, %eax
	movl	$32, %r10d
	movl	%r10d, %esi
	movabsq	$.L.str.17, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rcx
	movl	-180(%rbp), %r10d       # 4-byte Reload
	movl	%r10d, %ebx
	movl	%ebx, %edx
	movq	%rax, %rdi
	movq	%rcx, -192(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-192(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$19134, %r10d           # imm = 0x4ABE
	movl	%r10d, %edi
	movl	$32, %r10d
	movl	%r10d, %esi
	movl	$1, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.18, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rax
	movl	-180(%rbp), %r10d       # 4-byte Reload
	movl	-60(%rbp), %ebx         # 4-byte Reload
	imull	%ebx, %r10d
	.loc	1 6 15                  # /workspace/LLVM-Tracer/example/triad/triad.c:6:15
	movl	%r10d, %r14d
	movl	%r14d, %edx
	movq	%rax, (%rsp)
	movl	%r10d, -196(%rbp)       # 4-byte Spill
	callq	trace_logger_log_int
	movl	$6, %r10d
	movl	%r10d, %edi
	movabsq	$.L.str.4, %rsi
	movabsq	$.L.str.7, %rdx
	movabsq	$.L.str.20, %rcx
	movl	$11, %r10d
	movl	%r10d, %r8d
	movl	$1, %r9d
	movl	$1, (%rsp)
	callq	trace_logger_log0
	movl	$2, %r9d
	movl	%r9d, %edi
	movl	$32, %r9d
	movl	%r9d, %esi
	movl	$1, %r9d
	movl	%r9d, %ecx
	movabsq	$.L.str.14, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rax
	movl	-148(%rbp), %r10d       # 4-byte Reload
	movl	%r10d, %ebx
	movl	%ebx, %edx
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %r10d
	movl	%r10d, %eax
	movl	$32, %r10d
	movl	%r10d, %esi
	movabsq	$.L.str.18, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rcx
	movl	-196(%rbp), %r10d       # 4-byte Reload
	movl	%r10d, %ebx
	movl	%ebx, %edx
	movq	%rax, %rdi
	movq	%rcx, -208(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-208(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$19134, %r10d           # imm = 0x4ABE
	movl	%r10d, %edi
	movl	$32, %r10d
	movl	%r10d, %esi
	movl	$1, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.20, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rax
	movl	-196(%rbp), %r10d       # 4-byte Reload
	movl	-148(%rbp), %ebx        # 4-byte Reload
	addl	%ebx, %r10d
	.loc	1 6 3                   # /workspace/LLVM-Tracer/example/triad/triad.c:6:3
	movl	%r10d, %r14d
	movl	%r14d, %edx
	movq	%rax, (%rsp)
	movl	%r10d, -212(%rbp)       # 4-byte Spill
	callq	trace_logger_log_int
	movl	$6, %r10d
	movl	%r10d, %edi
	movabsq	$.L.str.4, %rsi
	movabsq	$.L.str.7, %rdx
	movabsq	$.L.str.21, %rcx
	movl	$32, %r10d
	movl	%r10d, %r8d
	movl	$1, %r9d
	movl	$1, (%rsp)
	callq	trace_logger_log0
	movl	$2, %r9d
	movl	%r9d, %edi
	movl	$64, %r9d
	movl	%r9d, %esi
	movl	$1, %r9d
	movl	%r9d, %ecx
	movabsq	$.L.str.9, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rax
	movq	-104(%rbp), %rdx        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %r10d
	movl	%r10d, %eax
	movl	$64, %r10d
	movl	%r10d, %esi
	movabsq	$.L.str.22, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rcx
	movq	%rax, %rdi
	movq	-56(%rbp), %rdx         # 8-byte Reload
	movq	%rcx, -224(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-224(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$19134, %r10d           # imm = 0x4ABE
	movl	%r10d, %edi
	movl	$64, %r10d
	movl	%r10d, %esi
	movl	$1, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.21, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rax
	movq	-104(%rbp), %rdx        # 8-byte Reload
	shlq	$2, %rdx
	movq	-56(%rbp), %r11         # 8-byte Reload
	addq	%rdx, %r11
	.loc	1 6 8                   # /workspace/LLVM-Tracer/example/triad/triad.c:6:8
	movq	%r11, %rdx
	movq	%rax, (%rsp)
	movq	%r11, -232(%rbp)        # 8-byte Spill
	callq	trace_logger_log_ptr
	movl	$6, %r10d
	movl	%r10d, %edi
	movabsq	$.L.str.4, %rsi
	movabsq	$.L.str.7, %rdx
	movabsq	$.L.str.23, %rcx
	movl	$31, %r10d
	movl	%r10d, %r8d
	movl	$1, %r9d
	movl	$1, (%rsp)
	callq	trace_logger_log0
	movl	$2, %r9d
	movl	%r9d, %edi
	movl	$64, %r9d
	movl	%r9d, %esi
	movl	$1, %r9d
	movl	%r9d, %ecx
	movabsq	$.L.str.21, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rax
	movq	-232(%rbp), %rdx        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$1, %r10d
	movl	%r10d, %eax
	movl	$32, %r10d
	movl	%r10d, %esi
	movabsq	$.L.str.20, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rcx
	movl	-212(%rbp), %r10d       # 4-byte Reload
	movl	%r10d, %ebx
	movl	%ebx, %edx
	movq	%rax, %rdi
	movq	%rcx, -240(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-240(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$5, %r10d
	movl	%r10d, %edi
	movabsq	$.L.str.4, %rsi
	movabsq	$.L.str.7, %rdx
	movabsq	$.L.str.10, %rcx
	movl	$11, %r10d
	movl	%r10d, %r8d
	movl	$1, %r9d
	movq	-56(%rbp), %rax         # 8-byte Reload
	movq	-104(%rbp), %r11        # 8-byte Reload
	movl	-212(%rbp), %r10d       # 4-byte Reload
	movl	%r10d, (%rax,%r11,4)
.Ltmp4:
	.loc	1 5 22 is_stmt 1        # /workspace/LLVM-Tracer/example/triad/triad.c:5:22
	movl	$1, (%rsp)
	callq	trace_logger_log0
	movl	$2, %r9d
	movl	%r9d, %edi
	movl	$64, %r9d
	movl	%r9d, %esi
	movl	$1, %r9d
	movl	%r9d, %edx
	xorl	%r9d, %r9d
	movl	%r9d, %eax
	movabsq	$.L.str.11, %r8
	movabsq	$.L.str.8, %rcx
	movq	%rcx, -248(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	%rax, %r9
	movq	-248(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %r10d
	movl	%r10d, %eax
	movl	$64, %r10d
	movl	%r10d, %esi
	movabsq	$.L.str.9, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rcx
	movq	%rax, %rdi
	movq	-104(%rbp), %rdx        # 8-byte Reload
	movq	%rcx, -256(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-256(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$19134, %r10d           # imm = 0x4ABE
	movl	%r10d, %edi
	movl	$64, %r10d
	movl	%r10d, %esi
	movl	$1, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.10, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rax
	movq	-104(%rbp), %rdx        # 8-byte Reload
	addq	$1, %rdx
	.loc	1 5 16 is_stmt 0        # /workspace/LLVM-Tracer/example/triad/triad.c:5:16
	movq	%rdx, -264(%rbp)        # 8-byte Spill
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$5, %r10d
	movl	%r10d, %edi
	movabsq	$.L.str.4, %rsi
	movabsq	$.L.str.7, %rdx
	movabsq	$.L.str.24, %rcx
	movl	$51, %r10d
	movl	%r10d, %r8d
	movl	$1, %r9d
	movl	$1, (%rsp)
	callq	trace_logger_log0
	movl	$2, %r9d
	movl	%r9d, %edi
	movl	$64, %r9d
	movl	%r9d, %esi
	movl	$2048, %r9d             # imm = 0x800
	movl	%r9d, %edx
	xorl	%r9d, %r9d
	movl	%r9d, %eax
	movabsq	$.L.str.11, %r8
	movabsq	$.L.str.8, %rcx
	movq	%rcx, -272(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	%rax, %r9
	movq	-272(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %r10d
	movl	%r10d, %eax
	movl	$64, %r10d
	movl	%r10d, %esi
	movabsq	$.L.str.10, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rcx
	movq	%rax, %rdi
	movq	-264(%rbp), %rdx        # 8-byte Reload
	movq	%rcx, -280(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-280(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$19134, %r10d           # imm = 0x4ABE
	movl	%r10d, %edi
	movl	$1, %r10d
	movl	%r10d, %eax
	movabsq	$.L.str.24, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rcx
	movq	-264(%rbp), %rdx        # 8-byte Reload
	cmpq	$2048, %rdx             # imm = 0x800
	sete	%r15b
.Ltmp5:
	.loc	1 5 7                   # /workspace/LLVM-Tracer/example/triad/triad.c:5:7
	movb	%r15b, %r12b
	andb	$1, %r12b
	movzbl	%r12b, %r10d
	movl	%r10d, %edx
	movq	%rax, %rsi
	movq	%rcx, -288(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-288(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	movb	%r15b, -289(%rbp)       # 1-byte Spill
	callq	trace_logger_log_int
	movl	$5, %r10d
	movl	%r10d, %edi
	movabsq	$.L.str.4, %rsi
	movabsq	$.L.str.7, %rdx
	movabsq	$.L.str.25, %rcx
	movl	$2, %r10d
	movl	%r10d, %r8d
	movl	$1, %r9d
	movl	$1, (%rsp)
	callq	trace_logger_log0
	movl	$3, %r9d
	movl	%r9d, %edi
	xorl	%r9d, %r9d
	movl	%r9d, %eax
	movl	$1, %r9d
	movl	%r9d, %ecx
	movabsq	$.L.str.26, %r8
	movabsq	$.L.str.8, %rdx
	movq	%rax, %rsi
	movq	%rdx, -304(%rbp)        # 8-byte Spill
	movq	%rax, %rdx
	movq	%rax, %r9
	movq	-304(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$2, %r10d
	movl	%r10d, %edi
	xorl	%r10d, %r10d
	movl	%r10d, %eax
	movl	$1, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.7, %r8
	movabsq	$.L.str.8, %rdx
	movq	%rax, %rsi
	movq	%rdx, -312(%rbp)        # 8-byte Spill
	movq	%rax, %rdx
	movq	%rax, %r9
	movq	-312(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %r10d
	movl	%r10d, %eax
	movabsq	$.L.str.24, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rcx
	movb	-289(%rbp), %r15b       # 1-byte Reload
	andb	$1, %r15b
	movzbl	%r15b, %r10d
	movl	%r10d, %edx
	movq	%rax, %rdi
	movq	%rax, %rsi
	movq	%rcx, -320(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-320(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movb	-289(%rbp), %r15b       # 1-byte Reload
	testb	$1, %r15b
	movq	-264(%rbp), %rax        # 8-byte Reload
	movq	%rax, -96(%rbp)         # 8-byte Spill
	jne	.LBB0_2
	jmp	.LBB0_1
.Ltmp6:
.LBB0_2:
	#DEBUG_VALUE: triad:i <- [DW_OP_constu 104, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: triad:s <- [DW_OP_constu 60, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: triad:c <- [DW_OP_constu 56, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: triad:b <- [DW_OP_constu 48, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: triad:a <- [DW_OP_constu 40, DW_OP_minus] [%rbp+0]
	.loc	1 0 7                   # /workspace/LLVM-Tracer/example/triad/triad.c:0:7
	movl	$8, %eax
	movl	%eax, %edi
	movabsq	$.L.str.4, %rsi
	movabsq	$.L.str.26, %rdx
	movabsq	$.L.str.27, %rcx
	movl	$1, %eax
	movl	%eax, %r8d
	movl	$1, %r9d
	.loc	1 8 1 is_stmt 1         # /workspace/LLVM-Tracer/example/triad/triad.c:8:1
	movl	$1, (%rsp)
	callq	trace_logger_log0
	movabsq	$.L.str.4, %rdi
	movl	$1, %eax
	movl	%eax, %esi
	movl	$1, %eax
	movl	%eax, %edx
	movl	%eax, %ecx
	callq	trace_logger_update_status
	addq	$304, %rsp              # imm = 0x130
	popq	%rbx
	popq	%r12
	popq	%r14
	popq	%r15
	popq	%rbp
	retq
.Ltmp7:
.Lfunc_end0:
	.size	triad, .Lfunc_end0-triad
	.cfi_endproc
                                        # -- End function
	.globl	main                    # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin1:
	.loc	1 10 0                  # /workspace/LLVM-Tracer/example/triad/triad.c:10:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r12
	pushq	%rbx
	subq	$1168, %rsp             # imm = 0x490
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	movl	$12, %eax
	movl	%eax, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.29, %rdx
	movabsq	$.L.str.30, %rcx
	movl	$54, %eax
	movl	%eax, %r8d
	xorl	%r9d, %r9d
	movl	$1, %eax
.Ltmp8:
	.loc	1 12 17 prologue_end    # /workspace/LLVM-Tracer/example/triad/triad.c:12:17
	movl	$1, (%rsp)
	movl	%eax, -36(%rbp)         # 4-byte Spill
	callq	trace_logger_log0
	movl	$2, %eax
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movabsq	$malloc, %rdx
	movl	$1, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.31, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %r10
	movq	%r10, (%rsp)
	callq	trace_logger_log_ptr
	movl	$1, %eax
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movl	$8192, %eax             # imm = 0x2000
	movl	%eax, %edx
	xorl	%eax, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.11, %r8
	movabsq	$.L.str.8, %r9
	movq	%rcx, -48(%rbp)         # 8-byte Spill
	movq	-48(%rbp), %r10         # 8-byte Reload
	movq	%r9, -56(%rbp)          # 8-byte Spill
	movq	%r10, %r9
	movq	-56(%rbp), %r11         # 8-byte Reload
	movq	%r11, (%rsp)
	callq	trace_logger_log_int
	movl	$8192, %eax             # imm = 0x2000
	movl	%eax, %edi
	callq	malloc
	movl	$19134, %ebx            # imm = 0x4ABE
	movl	%ebx, %edi
	movl	$64, %ebx
	movl	%ebx, %esi
	movl	$1, %ebx
	movl	%ebx, %ecx
	movabsq	$.L.str.30, %r8
	xorl	%ebx, %ebx
	movl	%ebx, %r9d
	movabsq	$.L.str.8, %rdx
	.loc	1 12 9 is_stmt 0        # /workspace/LLVM-Tracer/example/triad/triad.c:12:9
	movq	%rdx, -64(%rbp)         # 8-byte Spill
	movq	%rax, %rdx
	movq	-64(%rbp), %r10         # 8-byte Reload
	movq	%r10, (%rsp)
	movq	%rax, -72(%rbp)         # 8-byte Spill
	callq	trace_logger_log_ptr
	movl	$12, %ebx
	movl	%ebx, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.29, %rdx
	movabsq	$.L.str.32, %rcx
	movl	$47, %ebx
	movl	%ebx, %r8d
	xorl	%r9d, %r9d
	movl	$1, %ebx
	movl	$1, (%rsp)
	movl	%ebx, -76(%rbp)         # 4-byte Spill
	callq	trace_logger_log0
	movl	$1, %r9d
	movl	%r9d, %eax
	movl	$64, %r9d
	movl	%r9d, %esi
	movabsq	$.L.str.30, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rcx
	movq	%rax, %rdi
	movq	-72(%rbp), %rdx         # 8-byte Reload
	movq	%rcx, -88(%rbp)         # 8-byte Spill
	movq	%rax, %rcx
	movq	-88(%rbp), %rax         # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$19134, %ebx            # imm = 0x4ABE
	movl	%ebx, %edi
	movl	$64, %ebx
	movl	%ebx, %esi
	movl	$1, %ebx
	movl	%ebx, %ecx
	movabsq	$.L.str.32, %r8
	xorl	%ebx, %ebx
	movl	%ebx, %r9d
	movabsq	$.L.str.8, %rax
	movq	-72(%rbp), %rdx         # 8-byte Reload
	.loc	1 11 7 is_stmt 1        # /workspace/LLVM-Tracer/example/triad/triad.c:11:7
	movq	%rdx, -96(%rbp)         # 8-byte Spill
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$13, %ebx
	movl	%ebx, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.29, %rdx
	movabsq	$.L.str.33, %rcx
	movl	$54, %ebx
	movl	%ebx, %r8d
	xorl	%r9d, %r9d
	movl	$1, %ebx
.Ltmp9:
	#DEBUG_VALUE: main:a <- [DW_OP_constu 96, DW_OP_minus] [%rbp+0]
	.loc	1 13 17                 # /workspace/LLVM-Tracer/example/triad/triad.c:13:17
	movl	$1, (%rsp)
	movl	%ebx, -100(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$2, %r9d
	movl	%r9d, %edi
	movl	$64, %r9d
	movl	%r9d, %esi
	movabsq	$malloc, %rdx
	movl	$1, %r9d
	movl	%r9d, %ecx
	movabsq	$.L.str.31, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rax
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$1, %ebx
	movl	%ebx, %edi
	movl	$64, %ebx
	movl	%ebx, %esi
	movl	$8192, %ebx             # imm = 0x2000
	movl	%ebx, %edx
	xorl	%ebx, %ebx
	movl	%ebx, %eax
	movabsq	$.L.str.11, %r8
	movabsq	$.L.str.8, %rcx
	movq	%rcx, -112(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	%rax, %r9
	movq	-112(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$8192, %ebx             # imm = 0x2000
	movl	%ebx, %edi
	callq	malloc
	movl	$19134, %ebx            # imm = 0x4ABE
	movl	%ebx, %edi
	movl	$64, %ebx
	movl	%ebx, %esi
	movl	$1, %ebx
	movl	%ebx, %ecx
	movabsq	$.L.str.33, %r8
	xorl	%ebx, %ebx
	movl	%ebx, %r9d
	movabsq	$.L.str.8, %rdx
	.loc	1 13 9 is_stmt 0        # /workspace/LLVM-Tracer/example/triad/triad.c:13:9
	movq	%rdx, -120(%rbp)        # 8-byte Spill
	movq	%rax, %rdx
	movq	-120(%rbp), %r10        # 8-byte Reload
	movq	%r10, (%rsp)
	movq	%rax, -128(%rbp)        # 8-byte Spill
	callq	trace_logger_log_ptr
	movl	$13, %ebx
	movl	%ebx, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.29, %rdx
	movabsq	$.L.str.34, %rcx
	movl	$47, %ebx
	movl	%ebx, %r8d
	xorl	%r9d, %r9d
	movl	$1, %ebx
	movl	$1, (%rsp)
	movl	%ebx, -132(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$1, %r9d
	movl	%r9d, %eax
	movl	$64, %r9d
	movl	%r9d, %esi
	movabsq	$.L.str.33, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rcx
	movq	%rax, %rdi
	movq	-128(%rbp), %rdx        # 8-byte Reload
	movq	%rcx, -144(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-144(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$19134, %ebx            # imm = 0x4ABE
	movl	%ebx, %edi
	movl	$64, %ebx
	movl	%ebx, %esi
	movl	$1, %ebx
	movl	%ebx, %ecx
	movabsq	$.L.str.34, %r8
	xorl	%ebx, %ebx
	movl	%ebx, %r9d
	movabsq	$.L.str.8, %rax
	movq	-128(%rbp), %rdx        # 8-byte Reload
	.loc	1 11 11 is_stmt 1       # /workspace/LLVM-Tracer/example/triad/triad.c:11:11
	movq	%rdx, -152(%rbp)        # 8-byte Spill
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$14, %ebx
	movl	%ebx, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.29, %rdx
	movabsq	$.L.str.35, %rcx
	movl	$54, %ebx
	movl	%ebx, %r8d
	xorl	%r9d, %r9d
	movl	$1, %ebx
.Ltmp10:
	#DEBUG_VALUE: main:b <- [DW_OP_constu 152, DW_OP_minus] [%rbp+0]
	.loc	1 14 17                 # /workspace/LLVM-Tracer/example/triad/triad.c:14:17
	movl	$1, (%rsp)
	movl	%ebx, -156(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$2, %r9d
	movl	%r9d, %edi
	movl	$64, %r9d
	movl	%r9d, %esi
	movabsq	$malloc, %rdx
	movl	$1, %r9d
	movl	%r9d, %ecx
	movabsq	$.L.str.31, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rax
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$1, %ebx
	movl	%ebx, %edi
	movl	$64, %ebx
	movl	%ebx, %esi
	movl	$8192, %ebx             # imm = 0x2000
	movl	%ebx, %edx
	xorl	%ebx, %ebx
	movl	%ebx, %eax
	movabsq	$.L.str.11, %r8
	movabsq	$.L.str.8, %rcx
	movq	%rcx, -168(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	%rax, %r9
	movq	-168(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$8192, %ebx             # imm = 0x2000
	movl	%ebx, %edi
	callq	malloc
	movl	$19134, %ebx            # imm = 0x4ABE
	movl	%ebx, %edi
	movl	$64, %ebx
	movl	%ebx, %esi
	movl	$1, %ebx
	movl	%ebx, %ecx
	movabsq	$.L.str.35, %r8
	xorl	%ebx, %ebx
	movl	%ebx, %r9d
	movabsq	$.L.str.8, %rdx
	.loc	1 14 9 is_stmt 0        # /workspace/LLVM-Tracer/example/triad/triad.c:14:9
	movq	%rdx, -176(%rbp)        # 8-byte Spill
	movq	%rax, %rdx
	movq	-176(%rbp), %r10        # 8-byte Reload
	movq	%r10, (%rsp)
	movq	%rax, -184(%rbp)        # 8-byte Spill
	callq	trace_logger_log_ptr
	movl	$14, %ebx
	movl	%ebx, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.29, %rdx
	movabsq	$.L.str.36, %rcx
	movl	$47, %ebx
	movl	%ebx, %r8d
	xorl	%r9d, %r9d
	movl	$1, %ebx
	movl	$1, (%rsp)
	movl	%ebx, -188(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$1, %r9d
	movl	%r9d, %eax
	movl	$64, %r9d
	movl	%r9d, %esi
	movabsq	$.L.str.35, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rcx
	movq	%rax, %rdi
	movq	-184(%rbp), %rdx        # 8-byte Reload
	movq	%rcx, -200(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-200(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$19134, %ebx            # imm = 0x4ABE
	movl	%ebx, %edi
	movl	$64, %ebx
	movl	%ebx, %esi
	movl	$1, %ebx
	movl	%ebx, %ecx
	movabsq	$.L.str.36, %r8
	xorl	%ebx, %ebx
	movl	%ebx, %r9d
	movabsq	$.L.str.8, %rax
	movq	-184(%rbp), %rdx        # 8-byte Reload
	.loc	1 11 15 is_stmt 1       # /workspace/LLVM-Tracer/example/triad/triad.c:11:15
	movq	%rdx, -208(%rbp)        # 8-byte Spill
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$16, %ebx
	movl	%ebx, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.29, %rdx
	movabsq	$.L.str.12, %rcx
	movl	$54, %ebx
	movl	%ebx, %r8d
	xorl	%r9d, %r9d
	movl	$1, %ebx
.Ltmp11:
	#DEBUG_VALUE: main:c <- [DW_OP_constu 208, DW_OP_minus] [%rbp+0]
	.loc	1 16 9                  # /workspace/LLVM-Tracer/example/triad/triad.c:16:9
	movl	$1, (%rsp)
	movl	%ebx, -212(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$2, %r9d
	movl	%r9d, %edi
	movl	$64, %r9d
	movl	%r9d, %esi
	movabsq	$time, %rdx
	movl	$1, %r9d
	movl	%r9d, %ecx
	movabsq	$.L.str.37, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rax
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$1, %ebx
	movl	%ebx, %edi
	movl	$64, %ebx
	movl	%ebx, %esi
	xorl	%ebx, %ebx
	movl	%ebx, %eax
	movabsq	$.L.str.11, %r8
	movabsq	$.L.str.8, %rcx
	movq	%rax, %rdx
	movq	%rcx, -224(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	%rax, %r9
	movq	-224(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	xorl	%ebx, %ebx
	movl	%ebx, %edi
	callq	time
	movl	$19134, %ebx            # imm = 0x4ABE
	movl	%ebx, %edi
	movl	$64, %ebx
	movl	%ebx, %esi
	movl	$1, %ebx
	movl	%ebx, %ecx
	movabsq	$.L.str.12, %r8
	xorl	%ebx, %ebx
	movl	%ebx, %r9d
	movabsq	$.L.str.8, %rdx
	movq	%rdx, -232(%rbp)        # 8-byte Spill
	movq	%rax, %rdx
	movq	-232(%rbp), %r10        # 8-byte Reload
	movq	%r10, (%rsp)
	movq	%rax, -240(%rbp)        # 8-byte Spill
	callq	trace_logger_log_int
	movl	$16, %ebx
	movl	%ebx, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.29, %rdx
	movabsq	$.L.str.14, %rcx
	movl	$36, %ebx
	movl	%ebx, %r8d
	xorl	%r9d, %r9d
	movl	$1, %ebx
	movl	$1, (%rsp)
	movl	%ebx, -244(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$1, %r9d
	movl	%r9d, %eax
	movl	$64, %r9d
	movl	%r9d, %esi
	movabsq	$.L.str.12, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rcx
	movq	%rax, %rdi
	movq	-240(%rbp), %rdx        # 8-byte Reload
	movq	%rcx, -256(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-256(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$19134, %ebx            # imm = 0x4ABE
	movl	%ebx, %edi
	movl	$32, %ebx
	movl	%ebx, %esi
	movl	$1, %ebx
	movl	%ebx, %ecx
	movabsq	$.L.str.14, %r8
	xorl	%ebx, %ebx
	movl	%ebx, %r9d
	movabsq	$.L.str.8, %rax
	movq	-240(%rbp), %rdx        # 8-byte Reload
	movl	%edx, %ebx
	.loc	1 16 3 is_stmt 0        # /workspace/LLVM-Tracer/example/triad/triad.c:16:3
	movl	%ebx, %r14d
	movl	%r14d, %edx
	movq	%rax, (%rsp)
	movl	%ebx, -260(%rbp)        # 4-byte Spill
	callq	trace_logger_log_int
	movl	$16, %ebx
	movl	%ebx, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.29, %rdx
	movabsq	$.L.str.38, %rcx
	movl	$54, %ebx
	movl	%ebx, %r8d
	xorl	%r9d, %r9d
	movl	$1, %ebx
	movl	$1, (%rsp)
	movl	%ebx, -264(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$2, %r9d
	movl	%r9d, %edi
	movl	$64, %r9d
	movl	%r9d, %esi
	movabsq	$srand, %rdx
	movl	$1, %r9d
	movl	%r9d, %ecx
	movabsq	$.L.str.39, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rax
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$1, %ebx
	movl	%ebx, %eax
	movl	$32, %ebx
	movl	%ebx, %esi
	movabsq	$.L.str.14, %r8
	xorl	%ebx, %ebx
	movl	%ebx, %r9d
	movabsq	$.L.str.8, %rcx
	movl	-260(%rbp), %ebx        # 4-byte Reload
	movl	%ebx, %r14d
	movl	%r14d, %edx
	movq	%rax, %rdi
	movq	%rcx, -272(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-272(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	-260(%rbp), %edi        # 4-byte Reload
	callq	srand
	movl	$17, %edi
                                        # kill: def %rdi killed %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.29, %rdx
	movabsq	$.L.str.40, %rcx
	movl	$2, %ebx
	movl	%ebx, %r8d
	xorl	%r9d, %r9d
	movl	$1, %ebx
.Ltmp12:
	#DEBUG_VALUE: main:i <- 0
	.loc	1 17 7 is_stmt 1        # /workspace/LLVM-Tracer/example/triad/triad.c:17:7
	movl	$1, (%rsp)
	movl	%ebx, -276(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$1, %r9d
	movl	%r9d, %eax
	xorl	%r9d, %r9d
	movl	%r9d, %ecx
	movabsq	$.L.str.41, %r8
	movabsq	$.L.str.8, %rdx
	movq	%rax, %rdi
	movq	%rcx, %rsi
	movq	%rdx, -288(%rbp)        # 8-byte Spill
	movq	%rcx, %rdx
	movq	%rcx, -296(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-296(%rbp), %r9         # 8-byte Reload
	movq	-288(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	xorl	%ebx, %ebx
	movl	%ebx, %eax
	movq	%rax, -304(%rbp)        # 8-byte Spill
.Ltmp13:
.LBB1_1:                                # =>This Inner Loop Header: Depth=1
	#DEBUG_VALUE: main:c <- [DW_OP_constu 208, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: main:b <- [DW_OP_constu 152, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: main:a <- [DW_OP_constu 96, DW_OP_minus] [%rbp+0]
	.loc	1 0 7 is_stmt 0         # /workspace/LLVM-Tracer/example/triad/triad.c:0:7
	movq	-304(%rbp), %rax        # 8-byte Reload
	movq	$-1, %rdi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.41, %rdx
	movabsq	$.L.str.9, %rcx
	movl	$53, %r8d
                                        # kill: def %r8 killed %r8d
	xorl	%r9d, %r9d
	movl	$1, %r10d
	.loc	1 15 6 is_stmt 1        # /workspace/LLVM-Tracer/example/triad/triad.c:15:6
	movl	$1, (%rsp)
	movq	%rax, -312(%rbp)        # 8-byte Spill
	movl	%r10d, -316(%rbp)       # 4-byte Spill
	callq	trace_logger_log0
	movl	$2, %r9d
	movl	%r9d, %edi
	movl	$64, %r9d
	movl	%r9d, %esi
	xorl	%r9d, %r9d
	movl	%r9d, %edx
	movl	$1, %r9d
	movl	%r9d, %eax
	movabsq	$.L.str.42, %r8
	movabsq	$.L.str.41, %rcx
	movq	%rcx, -328(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	%rax, %r9
	movq	-328(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %r10d
	movl	%r10d, %eax
	movl	$64, %r10d
	movl	%r10d, %esi
	xorl	%r10d, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.11, %r8
	movabsq	$.L.str.29, %rdx
	movq	%rax, %rdi
	movq	%rdx, -336(%rbp)        # 8-byte Spill
	movq	%rcx, %rdx
	movq	%rax, %r9
	movq	-336(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$19134, %r10d           # imm = 0x4ABE
	movl	%r10d, %edi
	movl	$64, %r10d
	movl	%r10d, %esi
	movl	$1, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.9, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.29, %rax
	movq	-312(%rbp), %rdx        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$18, %r10d
	movl	%r10d, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.41, %rdx
	movabsq	$.L.str.18, %rcx
	movl	$32, %r10d
	movl	%r10d, %r8d
	xorl	%r9d, %r9d
	movl	$1, %r10d
.Ltmp14:
	#DEBUG_VALUE: main:i <- [DW_OP_constu 312, DW_OP_minus] [%rbp+0]
	.loc	1 18 3                  # /workspace/LLVM-Tracer/example/triad/triad.c:18:3
	movl	$1, (%rsp)
	movl	%r10d, -340(%rbp)       # 4-byte Spill
	callq	trace_logger_log0
	movl	$2, %r9d
	movl	%r9d, %edi
	movl	$64, %r9d
	movl	%r9d, %esi
	movl	$1, %r9d
	movl	%r9d, %ecx
	movabsq	$.L.str.9, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rax
	movq	-312(%rbp), %rdx        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %r10d
	movl	%r10d, %eax
	movl	$64, %r10d
	movl	%r10d, %esi
	movabsq	$.L.str.36, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rcx
	movq	%rax, %rdi
	movq	-208(%rbp), %rdx        # 8-byte Reload
	movq	%rcx, -352(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-352(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$19134, %r10d           # imm = 0x4ABE
	movl	%r10d, %edi
	movl	$64, %r10d
	movl	%r10d, %esi
	movl	$1, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.18, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rax
	movq	-312(%rbp), %rdx        # 8-byte Reload
	shlq	$2, %rdx
	movq	-208(%rbp), %r11        # 8-byte Reload
	addq	%rdx, %r11
	.loc	1 18 8 is_stmt 0        # /workspace/LLVM-Tracer/example/triad/triad.c:18:8
	movq	%r11, %rdx
	movq	%rax, (%rsp)
	movq	%r11, -360(%rbp)        # 8-byte Spill
	callq	trace_logger_log_ptr
	movl	$18, %r10d
	movl	%r10d, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.41, %rdx
	movabsq	$.L.str.43, %rcx
	movl	$31, %r10d
	movl	%r10d, %r8d
	xorl	%r9d, %r9d
	movl	$1, %r10d
	movl	$1, (%rsp)
	movl	%r10d, -364(%rbp)       # 4-byte Spill
	callq	trace_logger_log0
	movl	$2, %r9d
	movl	%r9d, %edi
	movl	$64, %r9d
	movl	%r9d, %esi
	movl	$1, %r9d
	movl	%r9d, %ecx
	movabsq	$.L.str.18, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rax
	movq	-360(%rbp), %rdx        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$1, %r10d
	movl	%r10d, %eax
	movl	$32, %r10d
	movl	%r10d, %esi
	xorl	%r10d, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.9, %r8
	movabsq	$.L.str.8, %rdx
	movq	%rax, %rdi
	movq	%rdx, -376(%rbp)        # 8-byte Spill
	movq	%rcx, %rdx
	movq	%rcx, -384(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-384(%rbp), %r9         # 8-byte Reload
	movq	-376(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$19, %r10d
	movl	%r10d, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.41, %rdx
	movabsq	$.L.str.20, %rcx
	movl	$54, %r10d
	movl	%r10d, %r8d
	xorl	%r9d, %r9d
	movl	$1, %r10d
	movq	-208(%rbp), %rax        # 8-byte Reload
	movq	-312(%rbp), %r11        # 8-byte Reload
	movl	$0, (%rax,%r11,4)
	.loc	1 19 10 is_stmt 1       # /workspace/LLVM-Tracer/example/triad/triad.c:19:10
	movl	$1, (%rsp)
	movl	%r10d, -388(%rbp)       # 4-byte Spill
	callq	trace_logger_log0
	movl	$1, %r9d
	movl	%r9d, %eax
	movl	$64, %r9d
	movl	%r9d, %esi
	movabsq	$rand, %rdx
	movabsq	$.L.str.44, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rcx
	movq	%rax, %rdi
	movq	%rcx, -400(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-400(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	callq	rand
	movl	$19134, %r10d           # imm = 0x4ABE
	movl	%r10d, %edi
	movl	$32, %r10d
	movl	%r10d, %esi
	movl	$1, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.20, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rdx
	.loc	1 19 3 is_stmt 0        # /workspace/LLVM-Tracer/example/triad/triad.c:19:3
	movl	%eax, %r10d
	movl	%r10d, %r11d
	movq	%rdx, -408(%rbp)        # 8-byte Spill
	movq	%r11, %rdx
	movq	-408(%rbp), %r11        # 8-byte Reload
	movq	%r11, (%rsp)
	movl	%eax, -412(%rbp)        # 4-byte Spill
	callq	trace_logger_log_int
	movl	$19, %eax
	movl	%eax, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.41, %rdx
	movabsq	$.L.str.21, %rcx
	movl	$32, %eax
	movl	%eax, %r8d
	xorl	%r9d, %r9d
	movl	$1, %eax
	movl	$1, (%rsp)
	movl	%eax, -416(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$2, %eax
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movl	$1, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.9, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	-312(%rbp), %r11        # 8-byte Reload
	movq	%rdx, -424(%rbp)        # 8-byte Spill
	movq	%r11, %rdx
	movq	-424(%rbp), %rbx        # 8-byte Reload
	movq	%rbx, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %eax
	movl	%eax, %ecx
	movl	$64, %eax
	movl	%eax, %esi
	movabsq	$.L.str.32, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	%rcx, %rdi
	movq	-96(%rbp), %r11         # 8-byte Reload
	movq	%rdx, -432(%rbp)        # 8-byte Spill
	movq	%r11, %rdx
	movq	-432(%rbp), %rbx        # 8-byte Reload
	movq	%rbx, (%rsp)
	callq	trace_logger_log_ptr
	movl	$19134, %eax            # imm = 0x4ABE
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movl	$1, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.21, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	-312(%rbp), %r11        # 8-byte Reload
	shlq	$2, %r11
	movq	-96(%rbp), %rbx         # 8-byte Reload
	addq	%r11, %rbx
	.loc	1 19 8                  # /workspace/LLVM-Tracer/example/triad/triad.c:19:8
	movq	%rdx, -440(%rbp)        # 8-byte Spill
	movq	%rbx, %rdx
	movq	-440(%rbp), %r11        # 8-byte Reload
	movq	%r11, (%rsp)
	movq	%rbx, -448(%rbp)        # 8-byte Spill
	callq	trace_logger_log_ptr
	movl	$19, %eax
	movl	%eax, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.41, %rdx
	movabsq	$.L.str.45, %rcx
	movl	$31, %eax
	movl	%eax, %r8d
	xorl	%r9d, %r9d
	movl	$1, %eax
	movl	$1, (%rsp)
	movl	%eax, -452(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$2, %eax
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movl	$1, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.21, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	-448(%rbp), %r11        # 8-byte Reload
	movq	%rdx, -464(%rbp)        # 8-byte Spill
	movq	%r11, %rdx
	movq	-464(%rbp), %rbx        # 8-byte Reload
	movq	%rbx, (%rsp)
	callq	trace_logger_log_ptr
	movl	$1, %eax
	movl	%eax, %ecx
	movl	$32, %eax
	movl	%eax, %esi
	movabsq	$.L.str.20, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movl	-412(%rbp), %eax        # 4-byte Reload
	movl	%eax, %r10d
	movl	%r10d, %edi
	movq	%rdi, -472(%rbp)        # 8-byte Spill
	movq	%rcx, %rdi
	movq	-472(%rbp), %r11        # 8-byte Reload
	movq	%rdx, -480(%rbp)        # 8-byte Spill
	movq	%r11, %rdx
	movq	-480(%rbp), %rbx        # 8-byte Reload
	movq	%rbx, (%rsp)
	callq	trace_logger_log_int
	movl	$20, %eax
	movl	%eax, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.41, %rdx
	movabsq	$.L.str.10, %rcx
	movl	$54, %eax
	movl	%eax, %r8d
	xorl	%r9d, %r9d
	movl	$1, %eax
	movq	-96(%rbp), %r11         # 8-byte Reload
	movq	-312(%rbp), %rbx        # 8-byte Reload
	movl	-412(%rbp), %r10d       # 4-byte Reload
	movl	%r10d, (%r11,%rbx,4)
	.loc	1 20 10 is_stmt 1       # /workspace/LLVM-Tracer/example/triad/triad.c:20:10
	movl	$1, (%rsp)
	movl	%eax, -484(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$1, %eax
	movl	%eax, %ecx
	movl	$64, %eax
	movl	%eax, %esi
	movabsq	$rand, %rdx
	movabsq	$.L.str.44, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdi
	movq	%rdi, -496(%rbp)        # 8-byte Spill
	movq	%rcx, %rdi
	movq	-496(%rbp), %r11        # 8-byte Reload
	movq	%r11, (%rsp)
	callq	trace_logger_log_ptr
	callq	rand
	movl	$19134, %r10d           # imm = 0x4ABE
	movl	%r10d, %edi
	movl	$32, %r10d
	movl	%r10d, %esi
	movl	$1, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.10, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rdx
	.loc	1 20 3 is_stmt 0        # /workspace/LLVM-Tracer/example/triad/triad.c:20:3
	movl	%eax, %r10d
	movl	%r10d, %r11d
	movq	%rdx, -504(%rbp)        # 8-byte Spill
	movq	%r11, %rdx
	movq	-504(%rbp), %r11        # 8-byte Reload
	movq	%r11, (%rsp)
	movl	%eax, -508(%rbp)        # 4-byte Spill
	callq	trace_logger_log_int
	movl	$20, %eax
	movl	%eax, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.41, %rdx
	movabsq	$.L.str.24, %rcx
	movl	$32, %eax
	movl	%eax, %r8d
	xorl	%r9d, %r9d
	movl	$1, %eax
	movl	$1, (%rsp)
	movl	%eax, -512(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$2, %eax
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movl	$1, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.9, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	-312(%rbp), %r11        # 8-byte Reload
	movq	%rdx, -520(%rbp)        # 8-byte Spill
	movq	%r11, %rdx
	movq	-520(%rbp), %rbx        # 8-byte Reload
	movq	%rbx, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %eax
	movl	%eax, %ecx
	movl	$64, %eax
	movl	%eax, %esi
	movabsq	$.L.str.34, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	%rcx, %rdi
	movq	-152(%rbp), %r11        # 8-byte Reload
	movq	%rdx, -528(%rbp)        # 8-byte Spill
	movq	%r11, %rdx
	movq	-528(%rbp), %rbx        # 8-byte Reload
	movq	%rbx, (%rsp)
	callq	trace_logger_log_ptr
	movl	$19134, %eax            # imm = 0x4ABE
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movl	$1, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.24, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	-312(%rbp), %r11        # 8-byte Reload
	shlq	$2, %r11
	movq	-152(%rbp), %rbx        # 8-byte Reload
	addq	%r11, %rbx
	.loc	1 20 8                  # /workspace/LLVM-Tracer/example/triad/triad.c:20:8
	movq	%rdx, -536(%rbp)        # 8-byte Spill
	movq	%rbx, %rdx
	movq	-536(%rbp), %r11        # 8-byte Reload
	movq	%r11, (%rsp)
	movq	%rbx, -544(%rbp)        # 8-byte Spill
	callq	trace_logger_log_ptr
	movl	$20, %eax
	movl	%eax, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.41, %rdx
	movabsq	$.L.str.46, %rcx
	movl	$31, %eax
	movl	%eax, %r8d
	xorl	%r9d, %r9d
	movl	$1, %eax
	movl	$1, (%rsp)
	movl	%eax, -548(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$2, %eax
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movl	$1, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.24, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	-544(%rbp), %r11        # 8-byte Reload
	movq	%rdx, -560(%rbp)        # 8-byte Spill
	movq	%r11, %rdx
	movq	-560(%rbp), %rbx        # 8-byte Reload
	movq	%rbx, (%rsp)
	callq	trace_logger_log_ptr
	movl	$1, %eax
	movl	%eax, %ecx
	movl	$32, %eax
	movl	%eax, %esi
	movabsq	$.L.str.10, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movl	-508(%rbp), %eax        # 4-byte Reload
	movl	%eax, %r10d
	movl	%r10d, %edi
	movq	%rdi, -568(%rbp)        # 8-byte Spill
	movq	%rcx, %rdi
	movq	-568(%rbp), %r11        # 8-byte Reload
	movq	%rdx, -576(%rbp)        # 8-byte Spill
	movq	%r11, %rdx
	movq	-576(%rbp), %rbx        # 8-byte Reload
	movq	%rbx, (%rsp)
	callq	trace_logger_log_int
	movl	$17, %eax
	movl	%eax, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.41, %rdx
	movabsq	$.L.str.42, %rcx
	movl	$11, %eax
	movl	%eax, %r8d
	xorl	%r9d, %r9d
	movl	$1, %eax
	movq	-152(%rbp), %r11        # 8-byte Reload
	movq	-312(%rbp), %rbx        # 8-byte Reload
	movl	-508(%rbp), %r10d       # 4-byte Reload
	movl	%r10d, (%r11,%rbx,4)
.Ltmp15:
	.loc	1 17 24 is_stmt 1       # /workspace/LLVM-Tracer/example/triad/triad.c:17:24
	movl	$1, (%rsp)
	movl	%eax, -580(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$2, %eax
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movl	$1, %eax
	movl	%eax, %edx
	xorl	%eax, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.11, %r8
	movabsq	$.L.str.8, %r11
	movq	%rcx, -592(%rbp)        # 8-byte Spill
	movq	-592(%rbp), %r9         # 8-byte Reload
	movq	%r11, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %eax
	movl	%eax, %ecx
	movl	$64, %eax
	movl	%eax, %esi
	movabsq	$.L.str.9, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	%rcx, %rdi
	movq	-312(%rbp), %r11        # 8-byte Reload
	movq	%rdx, -600(%rbp)        # 8-byte Spill
	movq	%r11, %rdx
	movq	-600(%rbp), %rbx        # 8-byte Reload
	movq	%rbx, (%rsp)
	callq	trace_logger_log_int
	movl	$19134, %eax            # imm = 0x4ABE
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movl	$1, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.42, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	-312(%rbp), %r11        # 8-byte Reload
	addq	$1, %r11
	.loc	1 17 17 is_stmt 0       # /workspace/LLVM-Tracer/example/triad/triad.c:17:17
	movq	%rdx, -608(%rbp)        # 8-byte Spill
	movq	%r11, %rdx
	movq	-608(%rbp), %rbx        # 8-byte Reload
	movq	%rbx, (%rsp)
	movq	%r11, -616(%rbp)        # 8-byte Spill
	callq	trace_logger_log_int
	movl	$17, %eax
	movl	%eax, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.41, %rdx
	movabsq	$.L.str.47, %rcx
	movl	$51, %eax
	movl	%eax, %r8d
	xorl	%r9d, %r9d
	movl	$1, %eax
	movl	$1, (%rsp)
	movl	%eax, -620(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$2, %eax
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movl	$2048, %eax             # imm = 0x800
	movl	%eax, %edx
	xorl	%eax, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.11, %r8
	movabsq	$.L.str.8, %r11
	movq	%rcx, -632(%rbp)        # 8-byte Spill
	movq	-632(%rbp), %r9         # 8-byte Reload
	movq	%r11, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %eax
	movl	%eax, %ecx
	movl	$64, %eax
	movl	%eax, %esi
	movabsq	$.L.str.42, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	%rcx, %rdi
	movq	-616(%rbp), %r11        # 8-byte Reload
	movq	%rdx, -640(%rbp)        # 8-byte Spill
	movq	%r11, %rdx
	movq	-640(%rbp), %rbx        # 8-byte Reload
	movq	%rbx, (%rsp)
	callq	trace_logger_log_int
	movl	$19134, %eax            # imm = 0x4ABE
	movl	%eax, %edi
	movl	$1, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.47, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	-616(%rbp), %rsi        # 8-byte Reload
	cmpq	$2048, %rsi             # imm = 0x800
	sete	%r14b
.Ltmp16:
	.loc	1 17 7                  # /workspace/LLVM-Tracer/example/triad/triad.c:17:7
	movb	%r14b, %r15b
	andb	$1, %r15b
	movzbl	%r15b, %eax
	movl	%eax, %r11d
	movq	%rcx, %rsi
	movq	%rdx, -648(%rbp)        # 8-byte Spill
	movq	%r11, %rdx
	movq	-648(%rbp), %r11        # 8-byte Reload
	movq	%r11, (%rsp)
	movb	%r14b, -649(%rbp)       # 1-byte Spill
	callq	trace_logger_log_int
	movl	$17, %eax
	movl	%eax, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.41, %rdx
	movabsq	$.L.str.48, %rcx
	movl	$2, %eax
	movl	%eax, %r8d
	xorl	%r9d, %r9d
	movl	$1, %eax
	movl	$1, (%rsp)
	movl	%eax, -656(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$3, %eax
	movl	%eax, %edi
	xorl	%eax, %eax
	movl	%eax, %ecx
	movl	$1, %eax
	movl	%eax, %edx
	movabsq	$.L.str.49, %r8
	movabsq	$.L.str.8, %rsi
	movq	%rsi, -664(%rbp)        # 8-byte Spill
	movq	%rcx, %rsi
	movq	%rdx, -672(%rbp)        # 8-byte Spill
	movq	%rcx, %rdx
	movq	-672(%rbp), %r11        # 8-byte Reload
	movq	%rcx, -680(%rbp)        # 8-byte Spill
	movq	%r11, %rcx
	movq	-680(%rbp), %r9         # 8-byte Reload
	movq	-664(%rbp), %rbx        # 8-byte Reload
	movq	%rbx, (%rsp)
	callq	trace_logger_log_int
	movl	$2, %eax
	movl	%eax, %edi
	xorl	%eax, %eax
	movl	%eax, %ecx
	movl	$1, %eax
	movl	%eax, %edx
	movabsq	$.L.str.41, %r8
	movabsq	$.L.str.8, %rsi
	movq	%rsi, -688(%rbp)        # 8-byte Spill
	movq	%rcx, %rsi
	movq	%rdx, -696(%rbp)        # 8-byte Spill
	movq	%rcx, %rdx
	movq	-696(%rbp), %r9         # 8-byte Reload
	movq	%rcx, -704(%rbp)        # 8-byte Spill
	movq	%r9, %rcx
	movq	-704(%rbp), %r9         # 8-byte Reload
	movq	-688(%rbp), %r11        # 8-byte Reload
	movq	%r11, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.47, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movb	-649(%rbp), %r14b       # 1-byte Reload
	andb	$1, %r14b
	movzbl	%r14b, %eax
	movl	%eax, %esi
	movq	%rcx, %rdi
	movq	%rsi, -712(%rbp)        # 8-byte Spill
	movq	%rcx, %rsi
	movq	-712(%rbp), %r11        # 8-byte Reload
	movq	%rdx, -720(%rbp)        # 8-byte Spill
	movq	%r11, %rdx
	movq	-720(%rbp), %rbx        # 8-byte Reload
	movq	%rbx, (%rsp)
	callq	trace_logger_log_int
	movb	-649(%rbp), %r14b       # 1-byte Reload
	testb	$1, %r14b
	movq	-616(%rbp), %rcx        # 8-byte Reload
	movq	%rcx, -304(%rbp)        # 8-byte Spill
	jne	.LBB1_2
	jmp	.LBB1_1
.Ltmp17:
.LBB1_2:
	#DEBUG_VALUE: main:i <- [DW_OP_constu 312, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: main:c <- [DW_OP_constu 208, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: main:b <- [DW_OP_constu 152, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: main:a <- [DW_OP_constu 96, DW_OP_minus] [%rbp+0]
	.loc	1 0 7                   # /workspace/LLVM-Tracer/example/triad/triad.c:0:7
	movl	$23, %eax
	movl	%eax, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.49, %rdx
	movabsq	$.L.str.50, %rcx
	movl	$54, %eax
	movl	%eax, %r8d
	xorl	%r9d, %r9d
	movl	$1, %eax
	.loc	1 23 2 is_stmt 1        # /workspace/LLVM-Tracer/example/triad/triad.c:23:2
	movl	$1, (%rsp)
	movl	%eax, -724(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$5, %eax
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movabsq	$triad, %rdx
	movl	$1, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.4, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %r10
	movq	%r10, (%rsp)
	callq	trace_logger_log_ptr
	movl	$1, %eax
	movl	%eax, %ecx
	movl	$64, %eax
	movl	%eax, %esi
	movabsq	$.L.str.32, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	%rcx, %rdi
	movq	-96(%rbp), %r10         # 8-byte Reload
	movq	%rdx, -736(%rbp)        # 8-byte Spill
	movq	%r10, %rdx
	movq	-736(%rbp), %r11        # 8-byte Reload
	movq	%r11, (%rsp)
	callq	trace_logger_log_ptr
	movl	$2, %eax
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movl	$1, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.34, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	-152(%rbp), %r10        # 8-byte Reload
	movq	%rdx, -744(%rbp)        # 8-byte Spill
	movq	%r10, %rdx
	movq	-744(%rbp), %r11        # 8-byte Reload
	movq	%r11, (%rsp)
	callq	trace_logger_log_ptr
	movl	$3, %eax
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movl	$1, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.36, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	-208(%rbp), %r10        # 8-byte Reload
	movq	%rdx, -752(%rbp)        # 8-byte Spill
	movq	%r10, %rdx
	movq	-752(%rbp), %r11        # 8-byte Reload
	movq	%r11, (%rsp)
	callq	trace_logger_log_ptr
	movl	$4, %eax
	movl	%eax, %edi
	movl	$32, %eax
	movl	%eax, %esi
	movl	$3, %eax
	movl	%eax, %edx
	xorl	%eax, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.11, %r8
	movabsq	$.L.str.8, %r9
	movq	%rcx, -760(%rbp)        # 8-byte Spill
	movq	-760(%rbp), %r10        # 8-byte Reload
	movq	%r9, -768(%rbp)         # 8-byte Spill
	movq	%r10, %r9
	movq	-768(%rbp), %r11        # 8-byte Reload
	movq	%r11, (%rsp)
	callq	trace_logger_log_int
	movl	$3, %ecx
	movq	-96(%rbp), %rdi         # 8-byte Reload
	movq	-152(%rbp), %rsi        # 8-byte Reload
	movq	-208(%rbp), %rdx        # 8-byte Reload
	callq	triad
	movl	$26, %eax
	movl	%eax, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.49, %rdx
	movabsq	$.L.str.51, %rcx
	movl	$54, %eax
	movl	%eax, %r8d
	xorl	%r9d, %r9d
	movl	$1, %eax
	.loc	1 26 12                 # /workspace/LLVM-Tracer/example/triad/triad.c:26:12
	movl	$1, (%rsp)
	movl	%eax, -772(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$3, %eax
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movabsq	$fopen, %rdx
	movl	$1, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.52, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %r10
	movq	%r10, (%rsp)
	callq	trace_logger_log_ptr
	movl	$1, %eax
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movabsq	$.L.str.53, %rdx
	xorl	%eax, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.11, %r8
	movabsq	$.L.str.8, %r9
	movq	%rcx, -784(%rbp)        # 8-byte Spill
	movq	-784(%rbp), %r10        # 8-byte Reload
	movq	%r9, -792(%rbp)         # 8-byte Spill
	movq	%r10, %r9
	movq	-792(%rbp), %r11        # 8-byte Reload
	movq	%r11, (%rsp)
	callq	trace_logger_log_string
	movl	$2, %eax
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movabsq	$.L.str.54, %rdx
	xorl	%eax, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.11, %r8
	movabsq	$.L.str.8, %r9
	movq	%rcx, -800(%rbp)        # 8-byte Spill
	movq	-800(%rbp), %r10        # 8-byte Reload
	movq	%r9, -808(%rbp)         # 8-byte Spill
	movq	%r10, %r9
	movq	-808(%rbp), %r11        # 8-byte Reload
	movq	%r11, (%rsp)
	callq	trace_logger_log_string
	movabsq	$.L.str, %rdi
	movabsq	$.L.str.1, %rsi
	callq	fopen
	movl	$19134, %ebx            # imm = 0x4ABE
	movl	%ebx, %edi
	movl	$64, %ebx
	movl	%ebx, %esi
	movl	$1, %ebx
	movl	%ebx, %ecx
	movabsq	$.L.str.51, %r8
	xorl	%ebx, %ebx
	movl	%ebx, %r9d
	movabsq	$.L.str.8, %rdx
	.loc	1 25 9                  # /workspace/LLVM-Tracer/example/triad/triad.c:25:9
	movq	%rdx, -816(%rbp)        # 8-byte Spill
	movq	%rax, %rdx
	movq	-816(%rbp), %r10        # 8-byte Reload
	movq	%r10, (%rsp)
	movq	%rax, -824(%rbp)        # 8-byte Spill
	callq	trace_logger_log_ptr
	movl	$27, %ebx
	movl	%ebx, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.49, %rdx
	movabsq	$.L.str.55, %rcx
	movl	$2, %ebx
	movl	%ebx, %r8d
	xorl	%r9d, %r9d
	movl	$1, %ebx
.Ltmp18:
	#DEBUG_VALUE: main:output <- [DW_OP_constu 824, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: main:i <- 0
	.loc	1 27 2                  # /workspace/LLVM-Tracer/example/triad/triad.c:27:2
	movl	$1, (%rsp)
	movl	%ebx, -828(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$1, %r9d
	movl	%r9d, %eax
	xorl	%r9d, %r9d
	movl	%r9d, %ecx
	movabsq	$.L.str.56, %r8
	movabsq	$.L.str.8, %rdx
	movq	%rax, %rdi
	movq	%rcx, %rsi
	movq	%rdx, -840(%rbp)        # 8-byte Spill
	movq	%rcx, %rdx
	movq	%rcx, -848(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-848(%rbp), %r9         # 8-byte Reload
	movq	-840(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	xorl	%ebx, %ebx
	movl	%ebx, %eax
	movq	%rax, -856(%rbp)        # 8-byte Spill
.Ltmp19:
.LBB1_3:                                # =>This Inner Loop Header: Depth=1
	#DEBUG_VALUE: main:output <- [DW_OP_constu 824, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: main:c <- [DW_OP_constu 208, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: main:b <- [DW_OP_constu 152, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: main:a <- [DW_OP_constu 96, DW_OP_minus] [%rbp+0]
	.loc	1 0 2 is_stmt 0         # /workspace/LLVM-Tracer/example/triad/triad.c:0:2
	movq	-856(%rbp), %rax        # 8-byte Reload
	movq	$-1, %rdi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.56, %rdx
	movabsq	$.L.str.9, %rcx
	movl	$53, %r8d
                                        # kill: def %r8 killed %r8d
	xorl	%r9d, %r9d
	movl	$1, %r10d
	.loc	1 15 6 is_stmt 1        # /workspace/LLVM-Tracer/example/triad/triad.c:15:6
	movl	$1, (%rsp)
	movq	%rax, -864(%rbp)        # 8-byte Spill
	movl	%r10d, -868(%rbp)       # 4-byte Spill
	callq	trace_logger_log0
	movl	$2, %r9d
	movl	%r9d, %edi
	movl	$64, %r9d
	movl	%r9d, %esi
	xorl	%r9d, %r9d
	movl	%r9d, %edx
	movl	$1, %r9d
	movl	%r9d, %eax
	movabsq	$.L.str.57, %r8
	movabsq	$.L.str.56, %rcx
	movq	%rcx, -880(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	%rax, %r9
	movq	-880(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %r10d
	movl	%r10d, %eax
	movl	$64, %r10d
	movl	%r10d, %esi
	xorl	%r10d, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.11, %r8
	movabsq	$.L.str.49, %rdx
	movq	%rax, %rdi
	movq	%rdx, -888(%rbp)        # 8-byte Spill
	movq	%rcx, %rdx
	movq	%rax, %r9
	movq	-888(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$19134, %r10d           # imm = 0x4ABE
	movl	%r10d, %edi
	movl	$64, %r10d
	movl	%r10d, %esi
	movl	$1, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.9, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.49, %rax
	movq	-864(%rbp), %rdx        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$28, %r10d
	movl	%r10d, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.56, %rdx
	movabsq	$.L.str.58, %rcx
	movl	$32, %r10d
	movl	%r10d, %r8d
	xorl	%r9d, %r9d
	movl	$1, %r10d
.Ltmp20:
	#DEBUG_VALUE: main:i <- [DW_OP_constu 864, DW_OP_minus] [%rbp+0]
	.loc	1 28 28                 # /workspace/LLVM-Tracer/example/triad/triad.c:28:28
	movl	$1, (%rsp)
	movl	%r10d, -892(%rbp)       # 4-byte Spill
	callq	trace_logger_log0
	movl	$2, %r9d
	movl	%r9d, %edi
	movl	$64, %r9d
	movl	%r9d, %esi
	movl	$1, %r9d
	movl	%r9d, %ecx
	movabsq	$.L.str.9, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rax
	movq	-864(%rbp), %rdx        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %r10d
	movl	%r10d, %eax
	movl	$64, %r10d
	movl	%r10d, %esi
	movabsq	$.L.str.36, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rcx
	movq	%rax, %rdi
	movq	-208(%rbp), %rdx        # 8-byte Reload
	movq	%rcx, -904(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-904(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$19134, %r10d           # imm = 0x4ABE
	movl	%r10d, %edi
	movl	$64, %r10d
	movl	%r10d, %esi
	movl	$1, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.58, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rax
	movq	-864(%rbp), %rdx        # 8-byte Reload
	shlq	$2, %rdx
	movq	-208(%rbp), %r11        # 8-byte Reload
	addq	%rdx, %r11
	movq	%r11, %rdx
	movq	%rax, (%rsp)
	movq	%r11, -912(%rbp)        # 8-byte Spill
	callq	trace_logger_log_ptr
	movl	$28, %r10d
	movl	%r10d, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.56, %rdx
	movabsq	$.L.str.59, %rcx
	movl	$30, %r10d
	movl	%r10d, %r8d
	xorl	%r9d, %r9d
	movl	$1, %r10d
	movl	$1, (%rsp)
	movl	%r10d, -916(%rbp)       # 4-byte Spill
	callq	trace_logger_log0
	movl	$1, %r9d
	movl	%r9d, %eax
	movl	$64, %r9d
	movl	%r9d, %esi
	movabsq	$.L.str.58, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rcx
	movq	%rax, %rdi
	movq	-912(%rbp), %rdx        # 8-byte Reload
	movq	%rcx, -928(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-928(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$19134, %r10d           # imm = 0x4ABE
	movl	%r10d, %edi
	movl	$32, %r10d
	movl	%r10d, %esi
	movl	$1, %r10d
	movl	%r10d, %ecx
	movabsq	$.L.str.59, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rax
	movq	-208(%rbp), %rdx        # 8-byte Reload
	movq	-864(%rbp), %r11        # 8-byte Reload
	movl	(%rdx,%r11,4), %r10d
	.loc	1 28 5 is_stmt 0        # /workspace/LLVM-Tracer/example/triad/triad.c:28:5
	movl	%r10d, %ebx
	movl	%ebx, %edx
	movq	%rax, (%rsp)
	movl	%r10d, -932(%rbp)       # 4-byte Spill
	callq	trace_logger_log_int
	movl	$28, %r10d
	movl	%r10d, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.56, %rdx
	movabsq	$.L.str.60, %rcx
	movl	$54, %r10d
	movl	%r10d, %r8d
	xorl	%r9d, %r9d
	movl	$1, %r10d
	movl	$1, (%rsp)
	movl	%r10d, -936(%rbp)       # 4-byte Spill
	callq	trace_logger_log0
	movl	$4, %r9d
	movl	%r9d, %edi
	movl	$64, %r9d
	movl	%r9d, %esi
	movabsq	$fprintf, %rdx
	movl	$1, %r9d
	movl	%r9d, %ecx
	movabsq	$.L.str.61, %r8
	xorl	%r9d, %r9d
                                        # kill: def %r9 killed %r9d
	movabsq	$.L.str.8, %rax
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$1, %r10d
	movl	%r10d, %eax
	movl	$64, %r10d
	movl	%r10d, %esi
	movabsq	$.L.str.51, %r8
	xorl	%r10d, %r10d
	movl	%r10d, %r9d
	movabsq	$.L.str.8, %rcx
	movq	%rax, %rdi
	movq	-824(%rbp), %rdx        # 8-byte Reload
	movq	%rcx, -944(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	-944(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_ptr
	movl	$2, %r10d
	movl	%r10d, %edi
	movl	$64, %r10d
	movl	%r10d, %esi
	movabsq	$.L.str.62, %rdx
	xorl	%r10d, %r10d
	movl	%r10d, %eax
	movabsq	$.L.str.11, %r8
	movabsq	$.L.str.8, %rcx
	movq	%rcx, -952(%rbp)        # 8-byte Spill
	movq	%rax, %rcx
	movq	%rax, %r9
	movq	-952(%rbp), %rax        # 8-byte Reload
	movq	%rax, (%rsp)
	callq	trace_logger_log_string
	movabsq	$.L.str.2, %rsi
	movq	-824(%rbp), %rdi        # 8-byte Reload
	movl	-932(%rbp), %edx        # 4-byte Reload
	movb	$0, %al
	callq	fprintf
	movl	$19134, %edx            # imm = 0x4ABE
	movl	%edx, %edi
	movl	$32, %edx
	movl	%edx, %esi
	movl	$1, %edx
	movl	%edx, %ecx
	movabsq	$.L.str.60, %r8
	xorl	%edx, %edx
	movl	%edx, %r9d
	movabsq	$.L.str.8, %r11
	.loc	1 27 19 is_stmt 1       # /workspace/LLVM-Tracer/example/triad/triad.c:27:19
	movl	%eax, %eax
	movl	%eax, %edx
	movq	%r11, (%rsp)
	callq	trace_logger_log_int
	movl	$27, %eax
	movl	%eax, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.56, %rdx
	movabsq	$.L.str.57, %rcx
	movl	$11, %eax
	movl	%eax, %r8d
	xorl	%r9d, %r9d
	movl	$1, %eax
	movl	$1, (%rsp)
	movl	%eax, -956(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$2, %eax
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movl	$1, %eax
	movl	%eax, %edx
	xorl	%eax, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.11, %r8
	movabsq	$.L.str.8, %r11
	movq	%rcx, -968(%rbp)        # 8-byte Spill
	movq	-968(%rbp), %r9         # 8-byte Reload
	movq	%r11, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %eax
	movl	%eax, %ecx
	movl	$64, %eax
	movl	%eax, %esi
	movabsq	$.L.str.9, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	%rcx, %rdi
	movq	-864(%rbp), %r11        # 8-byte Reload
	movq	%rdx, -976(%rbp)        # 8-byte Spill
	movq	%r11, %rdx
	movq	-976(%rbp), %r14        # 8-byte Reload
	movq	%r14, (%rsp)
	callq	trace_logger_log_int
	movl	$19134, %eax            # imm = 0x4ABE
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movl	$1, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.57, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	-864(%rbp), %r11        # 8-byte Reload
	addq	$1, %r11
	.loc	1 27 12 is_stmt 0       # /workspace/LLVM-Tracer/example/triad/triad.c:27:12
	movq	%rdx, -984(%rbp)        # 8-byte Spill
	movq	%r11, %rdx
	movq	-984(%rbp), %r14        # 8-byte Reload
	movq	%r14, (%rsp)
	movq	%r11, -992(%rbp)        # 8-byte Spill
	callq	trace_logger_log_int
	movl	$27, %eax
	movl	%eax, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.56, %rdx
	movabsq	$.L.str.63, %rcx
	movl	$51, %eax
	movl	%eax, %r8d
	xorl	%r9d, %r9d
	movl	$1, %eax
	movl	$1, (%rsp)
	movl	%eax, -996(%rbp)        # 4-byte Spill
	callq	trace_logger_log0
	movl	$2, %eax
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movl	$2048, %eax             # imm = 0x800
	movl	%eax, %edx
	xorl	%eax, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.11, %r8
	movabsq	$.L.str.8, %r11
	movq	%rcx, -1008(%rbp)       # 8-byte Spill
	movq	-1008(%rbp), %r9        # 8-byte Reload
	movq	%r11, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %eax
	movl	%eax, %ecx
	movl	$64, %eax
	movl	%eax, %esi
	movabsq	$.L.str.57, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	%rcx, %rdi
	movq	-992(%rbp), %r11        # 8-byte Reload
	movq	%rdx, -1016(%rbp)       # 8-byte Spill
	movq	%r11, %rdx
	movq	-1016(%rbp), %r14       # 8-byte Reload
	movq	%r14, (%rsp)
	callq	trace_logger_log_int
	movl	$19134, %eax            # imm = 0x4ABE
	movl	%eax, %edi
	movl	$1, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.63, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	-992(%rbp), %rsi        # 8-byte Reload
	cmpq	$2048, %rsi             # imm = 0x800
	sete	%r15b
.Ltmp21:
	.loc	1 27 2                  # /workspace/LLVM-Tracer/example/triad/triad.c:27:2
	movb	%r15b, %r12b
	andb	$1, %r12b
	movzbl	%r12b, %eax
	movl	%eax, %r11d
	movq	%rcx, %rsi
	movq	%rdx, -1024(%rbp)       # 8-byte Spill
	movq	%r11, %rdx
	movq	-1024(%rbp), %r11       # 8-byte Reload
	movq	%r11, (%rsp)
	movb	%r15b, -1025(%rbp)      # 1-byte Spill
	callq	trace_logger_log_int
	movl	$27, %eax
	movl	%eax, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.56, %rdx
	movabsq	$.L.str.64, %rcx
	movl	$2, %eax
	movl	%eax, %r8d
	xorl	%r9d, %r9d
	movl	$1, %eax
	movl	$1, (%rsp)
	movl	%eax, -1032(%rbp)       # 4-byte Spill
	callq	trace_logger_log0
	movl	$3, %eax
	movl	%eax, %edi
	xorl	%eax, %eax
	movl	%eax, %ecx
	movl	$1, %eax
	movl	%eax, %edx
	movabsq	$.L.str.65, %r8
	movabsq	$.L.str.8, %rsi
	movq	%rsi, -1040(%rbp)       # 8-byte Spill
	movq	%rcx, %rsi
	movq	%rdx, -1048(%rbp)       # 8-byte Spill
	movq	%rcx, %rdx
	movq	-1048(%rbp), %r11       # 8-byte Reload
	movq	%rcx, -1056(%rbp)       # 8-byte Spill
	movq	%r11, %rcx
	movq	-1056(%rbp), %r9        # 8-byte Reload
	movq	-1040(%rbp), %r14       # 8-byte Reload
	movq	%r14, (%rsp)
	callq	trace_logger_log_int
	movl	$2, %eax
	movl	%eax, %edi
	xorl	%eax, %eax
	movl	%eax, %ecx
	movl	$1, %eax
	movl	%eax, %edx
	movabsq	$.L.str.56, %r8
	movabsq	$.L.str.8, %rsi
	movq	%rsi, -1064(%rbp)       # 8-byte Spill
	movq	%rcx, %rsi
	movq	%rdx, -1072(%rbp)       # 8-byte Spill
	movq	%rcx, %rdx
	movq	-1072(%rbp), %r9        # 8-byte Reload
	movq	%rcx, -1080(%rbp)       # 8-byte Spill
	movq	%r9, %rcx
	movq	-1080(%rbp), %r9        # 8-byte Reload
	movq	-1064(%rbp), %r11       # 8-byte Reload
	movq	%r11, (%rsp)
	callq	trace_logger_log_int
	movl	$1, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.63, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movb	-1025(%rbp), %r15b      # 1-byte Reload
	andb	$1, %r15b
	movzbl	%r15b, %eax
	movl	%eax, %esi
	movq	%rcx, %rdi
	movq	%rsi, -1088(%rbp)       # 8-byte Spill
	movq	%rcx, %rsi
	movq	-1088(%rbp), %r11       # 8-byte Reload
	movq	%rdx, -1096(%rbp)       # 8-byte Spill
	movq	%r11, %rdx
	movq	-1096(%rbp), %r14       # 8-byte Reload
	movq	%r14, (%rsp)
	callq	trace_logger_log_int
	movb	-1025(%rbp), %r15b      # 1-byte Reload
	testb	$1, %r15b
	movq	-992(%rbp), %rcx        # 8-byte Reload
	movq	%rcx, -856(%rbp)        # 8-byte Spill
	jne	.LBB1_4
	jmp	.LBB1_3
.Ltmp22:
.LBB1_4:
	#DEBUG_VALUE: main:output <- [DW_OP_constu 824, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: main:i <- [DW_OP_constu 312, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: main:c <- [DW_OP_constu 208, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: main:b <- [DW_OP_constu 152, DW_OP_minus] [%rbp+0]
	#DEBUG_VALUE: main:a <- [DW_OP_constu 96, DW_OP_minus] [%rbp+0]
	.loc	1 0 2                   # /workspace/LLVM-Tracer/example/triad/triad.c:0:2
	movl	$29, %eax
	movl	%eax, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.65, %rdx
	movabsq	$.L.str.66, %rcx
	movl	$54, %eax
	movl	%eax, %r8d
	xorl	%r9d, %r9d
	movl	$1, %eax
	.loc	1 29 3 is_stmt 1        # /workspace/LLVM-Tracer/example/triad/triad.c:29:3
	movl	$1, (%rsp)
	movl	%eax, -1100(%rbp)       # 4-byte Spill
	callq	trace_logger_log0
	movl	$3, %eax
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movabsq	$fprintf, %rdx
	movl	$1, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.61, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %r10
	movq	%r10, (%rsp)
	callq	trace_logger_log_ptr
	movl	$1, %eax
	movl	%eax, %ecx
	movl	$64, %eax
	movl	%eax, %esi
	movabsq	$.L.str.51, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	%rcx, %rdi
	movq	-824(%rbp), %r10        # 8-byte Reload
	movq	%rdx, -1112(%rbp)       # 8-byte Spill
	movq	%r10, %rdx
	movq	-1112(%rbp), %r11       # 8-byte Reload
	movq	%r11, (%rsp)
	callq	trace_logger_log_ptr
	movl	$2, %eax
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movabsq	$.L.str.67, %rdx
	xorl	%eax, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.11, %r8
	movabsq	$.L.str.8, %r9
	movq	%rcx, -1120(%rbp)       # 8-byte Spill
	movq	-1120(%rbp), %r10       # 8-byte Reload
	movq	%r9, -1128(%rbp)        # 8-byte Spill
	movq	%r10, %r9
	movq	-1128(%rbp), %r11       # 8-byte Reload
	movq	%r11, (%rsp)
	callq	trace_logger_log_string
	movabsq	$.L.str.3, %rsi
	movq	-824(%rbp), %rdi        # 8-byte Reload
	movb	$0, %al
	callq	fprintf
	movl	$19134, %ebx            # imm = 0x4ABE
	movl	%ebx, %edi
	movl	$32, %ebx
	movl	%ebx, %esi
	movl	$1, %ebx
	movl	%ebx, %ecx
	movabsq	$.L.str.66, %r8
	xorl	%ebx, %ebx
	movl	%ebx, %r9d
	movabsq	$.L.str.8, %rdx
	.loc	1 30 3                  # /workspace/LLVM-Tracer/example/triad/triad.c:30:3
	movl	%eax, %eax
	movl	%eax, %r10d
	movq	%rdx, -1136(%rbp)       # 8-byte Spill
	movq	%r10, %rdx
	movq	-1136(%rbp), %r10       # 8-byte Reload
	movq	%r10, (%rsp)
	callq	trace_logger_log_int
	movl	$30, %eax
	movl	%eax, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.65, %rdx
	movabsq	$.L.str.68, %rcx
	movl	$54, %eax
	movl	%eax, %r8d
	xorl	%r9d, %r9d
	movl	$1, %eax
	movl	$1, (%rsp)
	movl	%eax, -1140(%rbp)       # 4-byte Spill
	callq	trace_logger_log0
	movl	$2, %eax
	movl	%eax, %edi
	movl	$64, %eax
	movl	%eax, %esi
	movabsq	$fclose, %rdx
	movl	$1, %eax
	movl	%eax, %ecx
	movabsq	$.L.str.69, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %r10
	movq	%r10, (%rsp)
	callq	trace_logger_log_ptr
	movl	$1, %eax
	movl	%eax, %ecx
	movl	$64, %eax
	movl	%eax, %esi
	movabsq	$.L.str.51, %r8
	xorl	%eax, %eax
	movl	%eax, %r9d
	movabsq	$.L.str.8, %rdx
	movq	%rcx, %rdi
	movq	-824(%rbp), %r10        # 8-byte Reload
	movq	%rdx, -1152(%rbp)       # 8-byte Spill
	movq	%r10, %rdx
	movq	-1152(%rbp), %r11       # 8-byte Reload
	movq	%r11, (%rsp)
	callq	trace_logger_log_ptr
	movq	-824(%rbp), %rdi        # 8-byte Reload
	callq	fclose
	movl	$19134, %ebx            # imm = 0x4ABE
	movl	%ebx, %edi
	movl	$32, %ebx
	movl	%ebx, %esi
	movl	$1, %ebx
	movl	%ebx, %ecx
	movabsq	$.L.str.68, %r8
	xorl	%ebx, %ebx
	movl	%ebx, %r9d
	movabsq	$.L.str.8, %rdx
	.loc	1 31 2                  # /workspace/LLVM-Tracer/example/triad/triad.c:31:2
	movl	%eax, %eax
	movl	%eax, %r10d
	movq	%rdx, -1160(%rbp)       # 8-byte Spill
	movq	%r10, %rdx
	movq	-1160(%rbp), %r10       # 8-byte Reload
	movq	%r10, (%rsp)
	callq	trace_logger_log_int
	movl	$31, %eax
	movl	%eax, %edi
	movabsq	$.L.str.28, %rsi
	movabsq	$.L.str.65, %rdx
	movabsq	$.L.str.70, %rcx
	movl	$1, %eax
	movl	%eax, %r8d
	xorl	%r9d, %r9d
	movl	$1, %eax
	movl	$1, (%rsp)
	movl	%eax, -1164(%rbp)       # 4-byte Spill
	callq	trace_logger_log0
	movl	$1, %eax
	movl	%eax, %ecx
	movl	$32, %eax
	movl	%eax, %esi
	xorl	%eax, %eax
	movl	%eax, %edx
	movabsq	$.L.str.9, %r8
	movabsq	$.L.str.8, %rdi
	movq	%rdi, -1176(%rbp)       # 8-byte Spill
	movq	%rcx, %rdi
	movq	%rdx, -1184(%rbp)       # 8-byte Spill
	movq	-1184(%rbp), %r9        # 8-byte Reload
	movq	-1176(%rbp), %r10       # 8-byte Reload
	movq	%r10, (%rsp)
	callq	trace_logger_log_int
	movabsq	$.L.str.28, %rdi
	movl	$1, %eax
	movl	%eax, %esi
	xorl	%edx, %edx
	movl	$1, %ecx
	callq	trace_logger_update_status
	xorl	%eax, %eax
	addq	$1168, %rsp             # imm = 0x490
	popq	%rbx
	popq	%r12
	popq	%r14
	popq	%r15
	popq	%rbp
	retq
.Ltmp23:
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.section	.text.startup,"ax",@progbits
	.p2align	4, 0x90         # -- Begin function _GLOBAL__sub_I_trace_logger.cpp
	.type	_GLOBAL__sub_I_trace_logger.cpp,@function
_GLOBAL__sub_I_trace_logger.cpp:        # @_GLOBAL__sub_I_trace_logger.cpp
.Lfunc_begin2:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	callq	__cxx_global_var_init
	popq	%rbp
	jmp	__cxx_global_var_init.1 # TAILCALL
.Lfunc_end2:
	.size	_GLOBAL__sub_I_trace_logger.cpp, .Lfunc_end2-_GLOBAL__sub_I_trace_logger.cpp
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90         # -- Begin function __cxx_global_var_init
	.type	__cxx_global_var_init,@function
__cxx_global_var_init:                  # @__cxx_global_var_init
.Lfunc_begin3:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	$_Z8gz_filesB5cxx11, %eax
	movl	%eax, %edi
	callq	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEEC2Ev
	movl	$__dtor__Z8gz_filesB5cxx11, %eax
	movl	%eax, %edi
	popq	%rbp
	jmp	atexit                  # TAILCALL
.Lfunc_end3:
	.size	__cxx_global_var_init, .Lfunc_end3-__cxx_global_var_init
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90         # -- Begin function __cxx_global_var_init.1
	.type	__cxx_global_var_init.1,@function
__cxx_global_var_init.1:                # @__cxx_global_var_init.1
.Lfunc_begin4:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	$_Z12labelmap_strB5cxx11, %eax
	movl	%eax, %edi
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2Ev
	movl	$__dtor__Z12labelmap_strB5cxx11, %eax
	movl	%eax, %edi
	popq	%rbp
	jmp	atexit                  # TAILCALL
.Lfunc_end4:
	.size	__cxx_global_var_init.1, .Lfunc_end4-__cxx_global_var_init.1
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90         # -- Begin function __dtor__Z12labelmap_strB5cxx11
	.type	__dtor__Z12labelmap_strB5cxx11,@function
__dtor__Z12labelmap_strB5cxx11:         # @__dtor__Z12labelmap_strB5cxx11
.Lfunc_begin5:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	$_Z12labelmap_strB5cxx11, %eax
	movl	%eax, %edi
	popq	%rbp
	jmp	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev # TAILCALL
.Lfunc_end5:
	.size	__dtor__Z12labelmap_strB5cxx11, .Lfunc_end5-__dtor__Z12labelmap_strB5cxx11
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEEC2Ev,"axG",@progbits,_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEEC2Ev,comdat
	.weak	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEEC2Ev # -- Begin function _ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEEC2Ev
	.p2align	4, 0x90
	.type	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEEC2Ev,@function
_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEEC2Ev: # @_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEEC2Ev
.Lfunc_begin6:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	jmp	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EEC2Ev # TAILCALL
.Lfunc_end6:
	.size	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEEC2Ev, .Lfunc_end6-_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEEC2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text.startup,"ax",@progbits
	.p2align	4, 0x90         # -- Begin function __dtor__Z8gz_filesB5cxx11
	.type	__dtor__Z8gz_filesB5cxx11,@function
__dtor__Z8gz_filesB5cxx11:              # @__dtor__Z8gz_filesB5cxx11
.Lfunc_begin7:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	$_Z8gz_filesB5cxx11, %eax
	movl	%eax, %edi
	popq	%rbp
	jmp	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEED2Ev # TAILCALL
.Lfunc_end7:
	.size	__dtor__Z8gz_filesB5cxx11, .Lfunc_end7-__dtor__Z8gz_filesB5cxx11
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEED2Ev,"axG",@progbits,_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEED2Ev,comdat
	.weak	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEED2Ev # -- Begin function _ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEED2Ev
	.p2align	4, 0x90
	.type	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEED2Ev,@function
_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEED2Ev: # @_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEED2Ev
.Lfunc_begin8:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	jmp	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EED2Ev # TAILCALL
.Lfunc_end8:
	.size	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEED2Ev, .Lfunc_end8-_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEED2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EED2Ev,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EED2Ev,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EED2Ev # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EED2Ev
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EED2Ev,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EED2Ev: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EED2Ev
.Lfunc_begin9:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception0
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_beginEv
.Ltmp24:
	movq	-8(%rbp), %rdi          # 8-byte Reload
	movq	%rax, %rsi
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_eraseEPSt13_Rb_tree_nodeISA_E
.Ltmp25:
	jmp	.LBB9_1
.LBB9_1:
	movq	-8(%rbp), %rax          # 8-byte Reload
	movq	%rax, %rdi
	addq	$32, %rsp
	popq	%rbp
	jmp	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEED2Ev # TAILCALL
.LBB9_2:
.Ltmp26:
	movl	%edx, %ecx
	movq	-8(%rbp), %rdx          # 8-byte Reload
	movq	%rdx, %rdi
	movl	%ecx, -12(%rbp)         # 4-byte Spill
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEED2Ev
	movq	-24(%rbp), %rdi         # 8-byte Reload
	callq	__clang_call_terminate
.Lfunc_end9:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EED2Ev, .Lfunc_end9-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EED2Ev
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2
GCC_except_table9:
.Lexception0:
	.byte	255                     # @LPStart Encoding = omit
	.byte	3                       # @TType Encoding = udata4
	.byte	21                      # @TType base offset
	.byte	3                       # Call site Encoding = udata4
	.byte	13                      # Call site table length
	.long	.Ltmp24-.Lfunc_begin9   # >> Call Site 1 <<
	.long	.Ltmp25-.Ltmp24         #   Call between .Ltmp24 and .Ltmp25
	.long	.Ltmp26-.Lfunc_begin9   #     jumps to .Ltmp26
	.byte	1                       #   On action: 1
	.byte	1                       # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                       #   No further actions
                                        # >> Catch TypeInfos <<
	.long	0                       # TypeInfo 1
	.p2align	2
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_beginEv,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_beginEv,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_beginEv # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_beginEv
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_beginEv,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_beginEv: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_beginEv
.Lfunc_begin10:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	16(%rdi), %rax
	popq	%rbp
	retq
.Lfunc_end10:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_beginEv, .Lfunc_end10-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_beginEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_eraseEPSt13_Rb_tree_nodeISA_E,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_eraseEPSt13_Rb_tree_nodeISA_E,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_eraseEPSt13_Rb_tree_nodeISA_E # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_eraseEPSt13_Rb_tree_nodeISA_E
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_eraseEPSt13_Rb_tree_nodeISA_E,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_eraseEPSt13_Rb_tree_nodeISA_E: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_eraseEPSt13_Rb_tree_nodeISA_E
.Lfunc_begin11:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	cmpq	$0, %rsi
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	movq	%rsi, -16(%rbp)         # 8-byte Spill
	je	.LBB11_3
# %bb.1:
	movq	-16(%rbp), %rax         # 8-byte Reload
	movq	%rax, -24(%rbp)         # 8-byte Spill
	jmp	.LBB11_2
.LBB11_2:                               # =>This Inner Loop Header: Depth=1
	movq	-24(%rbp), %rax         # 8-byte Reload
	movq	%rax, %rdi
	movq	%rax, -32(%rbp)         # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_rightEPSt18_Rb_tree_node_base
	movq	-8(%rbp), %rdi          # 8-byte Reload
	movq	%rax, %rsi
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_eraseEPSt13_Rb_tree_nodeISA_E
	movq	-32(%rbp), %rdi         # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE7_S_leftEPSt18_Rb_tree_node_base
	movq	-8(%rbp), %rdi          # 8-byte Reload
	movq	-32(%rbp), %rsi         # 8-byte Reload
	movq	%rax, -40(%rbp)         # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_drop_nodeEPSt13_Rb_tree_nodeISA_E
	movq	-40(%rbp), %rax         # 8-byte Reload
	cmpq	$0, %rax
	movq	%rax, -24(%rbp)         # 8-byte Spill
	jne	.LBB11_2
.LBB11_3:
	addq	$48, %rsp
	popq	%rbp
	retq
.Lfunc_end11:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_eraseEPSt13_Rb_tree_nodeISA_E, .Lfunc_end11-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_eraseEPSt13_Rb_tree_nodeISA_E
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEED2Ev,"axG",@progbits,_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEED2Ev,comdat
	.weak	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEED2Ev # -- Begin function _ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEED2Ev
	.p2align	4, 0x90
	.type	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEED2Ev,@function
_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEED2Ev: # @_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEED2Ev
.Lfunc_begin12:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	popq	%rbp
	retq
.Lfunc_end12:
	.size	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEED2Ev, .Lfunc_end12-_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEED2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text.__clang_call_terminate,"axG",@progbits,__clang_call_terminate,comdat
	.hidden	__clang_call_terminate  # -- Begin function __clang_call_terminate
	.weak	__clang_call_terminate
	.p2align	4, 0x90
	.type	__clang_call_terminate,@function
__clang_call_terminate:                 # @__clang_call_terminate
.Lfunc_begin13:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	callq	__cxa_begin_catch
	movq	%rax, -8(%rbp)          # 8-byte Spill
	callq	_ZSt9terminatev
.Lfunc_end13:
	.size	__clang_call_terminate, .Lfunc_end13-__clang_call_terminate
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_rightEPSt18_Rb_tree_node_base,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_rightEPSt18_Rb_tree_node_base,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_rightEPSt18_Rb_tree_node_base # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_rightEPSt18_Rb_tree_node_base
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_rightEPSt18_Rb_tree_node_base,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_rightEPSt18_Rb_tree_node_base: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_rightEPSt18_Rb_tree_node_base
.Lfunc_begin14:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	24(%rdi), %rax
	popq	%rbp
	retq
.Lfunc_end14:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_rightEPSt18_Rb_tree_node_base, .Lfunc_end14-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_rightEPSt18_Rb_tree_node_base
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE7_S_leftEPSt18_Rb_tree_node_base,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE7_S_leftEPSt18_Rb_tree_node_base,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE7_S_leftEPSt18_Rb_tree_node_base # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE7_S_leftEPSt18_Rb_tree_node_base
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE7_S_leftEPSt18_Rb_tree_node_base,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE7_S_leftEPSt18_Rb_tree_node_base: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE7_S_leftEPSt18_Rb_tree_node_base
.Lfunc_begin15:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	16(%rdi), %rax
	popq	%rbp
	retq
.Lfunc_end15:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE7_S_leftEPSt18_Rb_tree_node_base, .Lfunc_end15-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE7_S_leftEPSt18_Rb_tree_node_base
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_drop_nodeEPSt13_Rb_tree_nodeISA_E,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_drop_nodeEPSt13_Rb_tree_nodeISA_E,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_drop_nodeEPSt13_Rb_tree_nodeISA_E # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_drop_nodeEPSt13_Rb_tree_nodeISA_E
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_drop_nodeEPSt13_Rb_tree_nodeISA_E,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_drop_nodeEPSt13_Rb_tree_nodeISA_E: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_drop_nodeEPSt13_Rb_tree_nodeISA_E
.Lfunc_begin16:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	movq	%rsi, -16(%rbp)         # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE15_M_destroy_nodeEPSt13_Rb_tree_nodeISA_E
	movq	-8(%rbp), %rdi          # 8-byte Reload
	movq	-16(%rbp), %rsi         # 8-byte Reload
	addq	$16, %rsp
	popq	%rbp
	jmp	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_put_nodeEPSt13_Rb_tree_nodeISA_E # TAILCALL
.Lfunc_end16:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_drop_nodeEPSt13_Rb_tree_nodeISA_E, .Lfunc_end16-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_drop_nodeEPSt13_Rb_tree_nodeISA_E
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE15_M_destroy_nodeEPSt13_Rb_tree_nodeISA_E,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE15_M_destroy_nodeEPSt13_Rb_tree_nodeISA_E,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE15_M_destroy_nodeEPSt13_Rb_tree_nodeISA_E # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE15_M_destroy_nodeEPSt13_Rb_tree_nodeISA_E
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE15_M_destroy_nodeEPSt13_Rb_tree_nodeISA_E,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE15_M_destroy_nodeEPSt13_Rb_tree_nodeISA_E: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE15_M_destroy_nodeEPSt13_Rb_tree_nodeISA_E
.Lfunc_begin17:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception1
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rsi, -8(%rbp)          # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE21_M_get_Node_allocatorEv
	movq	-8(%rbp), %rdi          # 8-byte Reload
	movq	%rax, -16(%rbp)         # 8-byte Spill
	callq	_ZNSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv
.Ltmp27:
	movq	-16(%rbp), %rdi         # 8-byte Reload
	movq	%rax, %rsi
	callq	_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE7destroyISB_EEvRSD_PT_
.Ltmp28:
	jmp	.LBB17_1
.LBB17_1:
	addq	$32, %rsp
	popq	%rbp
	retq
.LBB17_2:
.Ltmp29:
	movl	%edx, %ecx
	movq	%rax, %rdi
	movl	%ecx, -20(%rbp)         # 4-byte Spill
	callq	__clang_call_terminate
.Lfunc_end17:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE15_M_destroy_nodeEPSt13_Rb_tree_nodeISA_E, .Lfunc_end17-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE15_M_destroy_nodeEPSt13_Rb_tree_nodeISA_E
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2
GCC_except_table17:
.Lexception1:
	.byte	255                     # @LPStart Encoding = omit
	.byte	3                       # @TType Encoding = udata4
	.byte	21                      # @TType base offset
	.byte	3                       # Call site Encoding = udata4
	.byte	13                      # Call site table length
	.long	.Ltmp27-.Lfunc_begin17  # >> Call Site 1 <<
	.long	.Ltmp28-.Ltmp27         #   Call between .Ltmp27 and .Ltmp28
	.long	.Ltmp29-.Lfunc_begin17  #     jumps to .Ltmp29
	.byte	1                       #   On action: 1
	.byte	1                       # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                       #   No further actions
                                        # >> Catch TypeInfos <<
	.long	0                       # TypeInfo 1
	.p2align	2
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_put_nodeEPSt13_Rb_tree_nodeISA_E,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_put_nodeEPSt13_Rb_tree_nodeISA_E,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_put_nodeEPSt13_Rb_tree_nodeISA_E # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_put_nodeEPSt13_Rb_tree_nodeISA_E
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_put_nodeEPSt13_Rb_tree_nodeISA_E,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_put_nodeEPSt13_Rb_tree_nodeISA_E: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_put_nodeEPSt13_Rb_tree_nodeISA_E
.Lfunc_begin18:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception2
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rsi, -8(%rbp)          # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE21_M_get_Node_allocatorEv
.Ltmp30:
	movl	$1, %ecx
	movl	%ecx, %edx
	movq	%rax, %rdi
	movq	-8(%rbp), %rsi          # 8-byte Reload
	callq	_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE10deallocateERSD_PSC_m
.Ltmp31:
	jmp	.LBB18_1
.LBB18_1:
	addq	$16, %rsp
	popq	%rbp
	retq
.LBB18_2:
.Ltmp32:
	movl	%edx, %ecx
	movq	%rax, %rdi
	movl	%ecx, -12(%rbp)         # 4-byte Spill
	callq	__clang_call_terminate
.Lfunc_end18:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_put_nodeEPSt13_Rb_tree_nodeISA_E, .Lfunc_end18-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_put_nodeEPSt13_Rb_tree_nodeISA_E
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2
GCC_except_table18:
.Lexception2:
	.byte	255                     # @LPStart Encoding = omit
	.byte	3                       # @TType Encoding = udata4
	.byte	21                      # @TType base offset
	.byte	3                       # Call site Encoding = udata4
	.byte	13                      # Call site table length
	.long	.Ltmp30-.Lfunc_begin18  # >> Call Site 1 <<
	.long	.Ltmp31-.Ltmp30         #   Call between .Ltmp30 and .Ltmp31
	.long	.Ltmp32-.Lfunc_begin18  #     jumps to .Ltmp32
	.byte	1                       #   On action: 1
	.byte	1                       # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                       #   No further actions
                                        # >> Catch TypeInfos <<
	.long	0                       # TypeInfo 1
	.p2align	2
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE21_M_get_Node_allocatorEv,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE21_M_get_Node_allocatorEv,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE21_M_get_Node_allocatorEv # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE21_M_get_Node_allocatorEv
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE21_M_get_Node_allocatorEv,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE21_M_get_Node_allocatorEv: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE21_M_get_Node_allocatorEv
.Lfunc_begin19:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, %rax
	popq	%rbp
	retq
.Lfunc_end19:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE21_M_get_Node_allocatorEv, .Lfunc_end19-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE21_M_get_Node_allocatorEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE10deallocateERSD_PSC_m,"axG",@progbits,_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE10deallocateERSD_PSC_m,comdat
	.weak	_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE10deallocateERSD_PSC_m # -- Begin function _ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE10deallocateERSD_PSC_m
	.p2align	4, 0x90
	.type	_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE10deallocateERSD_PSC_m,@function
_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE10deallocateERSD_PSC_m: # @_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE10deallocateERSD_PSC_m
.Lfunc_begin20:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	jmp	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE10deallocateEPSD_m # TAILCALL
.Lfunc_end20:
	.size	_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE10deallocateERSD_PSC_m, .Lfunc_end20-_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE10deallocateERSD_PSC_m
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE10deallocateEPSD_m,"axG",@progbits,_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE10deallocateEPSD_m,comdat
	.weak	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE10deallocateEPSD_m # -- Begin function _ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE10deallocateEPSD_m
	.p2align	4, 0x90
	.type	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE10deallocateEPSD_m,@function
_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE10deallocateEPSD_m: # @_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE10deallocateEPSD_m
.Lfunc_begin21:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	movq	%rsi, %rdi
	movq	%rdx, -16(%rbp)         # 8-byte Spill
	popq	%rbp
	jmp	_ZdlPv                  # TAILCALL
.Lfunc_end21:
	.size	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE10deallocateEPSD_m, .Lfunc_end21-_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE10deallocateEPSD_m
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv,"axG",@progbits,_ZNSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv,comdat
	.weak	_ZNSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv # -- Begin function _ZNSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv
	.p2align	4, 0x90
	.type	_ZNSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv,@function
_ZNSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv: # @_ZNSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv
.Lfunc_begin22:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	addq	$32, %rdi
	popq	%rbp
	jmp	_ZN9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE6_M_ptrEv # TAILCALL
.Lfunc_end22:
	.size	_ZNSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv, .Lfunc_end22-_ZNSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE7destroyISB_EEvRSD_PT_,"axG",@progbits,_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE7destroyISB_EEvRSD_PT_,comdat
	.weak	_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE7destroyISB_EEvRSD_PT_ # -- Begin function _ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE7destroyISB_EEvRSD_PT_
	.p2align	4, 0x90
	.type	_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE7destroyISB_EEvRSD_PT_,@function
_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE7destroyISB_EEvRSD_PT_: # @_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE7destroyISB_EEvRSD_PT_
.Lfunc_begin23:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	jmp	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE7destroyISC_EEvPT_ # TAILCALL
.Lfunc_end23:
	.size	_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE7destroyISB_EEvRSD_PT_, .Lfunc_end23-_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE7destroyISB_EEvRSD_PT_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE7destroyISC_EEvPT_,"axG",@progbits,_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE7destroyISC_EEvPT_,comdat
	.weak	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE7destroyISC_EEvPT_ # -- Begin function _ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE7destroyISC_EEvPT_
	.p2align	4, 0x90
	.type	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE7destroyISC_EEvPT_,@function
_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE7destroyISC_EEvPT_: # @_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE7destroyISC_EEvPT_
.Lfunc_begin24:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	movq	%rsi, %rdi
	popq	%rbp
	jmp	_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sED2Ev # TAILCALL
.Lfunc_end24:
	.size	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE7destroyISC_EEvPT_, .Lfunc_end24-_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE7destroyISC_EEvPT_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sED2Ev,"axG",@progbits,_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sED2Ev,comdat
	.weak	_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sED2Ev # -- Begin function _ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sED2Ev
	.p2align	4, 0x90
	.type	_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sED2Ev,@function
_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sED2Ev: # @_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sED2Ev
.Lfunc_begin25:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	jmp	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev # TAILCALL
.Lfunc_end25:
	.size	_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sED2Ev, .Lfunc_end25-_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sED2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE6_M_ptrEv,"axG",@progbits,_ZN9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE6_M_ptrEv,comdat
	.weak	_ZN9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE6_M_ptrEv # -- Begin function _ZN9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE6_M_ptrEv
	.p2align	4, 0x90
	.type	_ZN9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE6_M_ptrEv,@function
_ZN9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE6_M_ptrEv: # @_ZN9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE6_M_ptrEv
.Lfunc_begin26:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	jmp	_ZN9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE7_M_addrEv # TAILCALL
.Lfunc_end26:
	.size	_ZN9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE6_M_ptrEv, .Lfunc_end26-_ZN9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE6_M_ptrEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE7_M_addrEv,"axG",@progbits,_ZN9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE7_M_addrEv,comdat
	.weak	_ZN9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE7_M_addrEv # -- Begin function _ZN9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE7_M_addrEv
	.p2align	4, 0x90
	.type	_ZN9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE7_M_addrEv,@function
_ZN9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE7_M_addrEv: # @_ZN9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE7_M_addrEv
.Lfunc_begin27:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, %rax
	popq	%rbp
	retq
.Lfunc_end27:
	.size	_ZN9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE7_M_addrEv, .Lfunc_end27-_ZN9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE7_M_addrEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EEC2Ev,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EEC2Ev,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EEC2Ev # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EEC2Ev
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EEC2Ev,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EEC2Ev: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EEC2Ev
.Lfunc_begin28:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	jmp	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE13_Rb_tree_implISE_Lb1EEC2Ev # TAILCALL
.Lfunc_end28:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EEC2Ev, .Lfunc_end28-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EEC2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE13_Rb_tree_implISE_Lb1EEC2Ev,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE13_Rb_tree_implISE_Lb1EEC2Ev,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE13_Rb_tree_implISE_Lb1EEC2Ev # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE13_Rb_tree_implISE_Lb1EEC2Ev
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE13_Rb_tree_implISE_Lb1EEC2Ev,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE13_Rb_tree_implISE_Lb1EEC2Ev: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE13_Rb_tree_implISE_Lb1EEC2Ev
.Lfunc_begin29:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	callq	_ZNSaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEC2Ev
	movq	-8(%rbp), %rdi          # 8-byte Reload
	callq	_ZNSt20_Rb_tree_key_compareISt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2Ev
	movq	-8(%rbp), %rdi          # 8-byte Reload
	addq	$8, %rdi
	addq	$16, %rsp
	popq	%rbp
	jmp	_ZNSt15_Rb_tree_headerC2Ev # TAILCALL
.Lfunc_end29:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE13_Rb_tree_implISE_Lb1EEC2Ev, .Lfunc_end29-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE13_Rb_tree_implISE_Lb1EEC2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEC2Ev,"axG",@progbits,_ZNSaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEC2Ev,comdat
	.weak	_ZNSaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEC2Ev # -- Begin function _ZNSaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEC2Ev
	.p2align	4, 0x90
	.type	_ZNSaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEC2Ev,@function
_ZNSaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEC2Ev: # @_ZNSaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEC2Ev
.Lfunc_begin30:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	jmp	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEC2Ev # TAILCALL
.Lfunc_end30:
	.size	_ZNSaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEC2Ev, .Lfunc_end30-_ZNSaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEC2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt20_Rb_tree_key_compareISt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2Ev,"axG",@progbits,_ZNSt20_Rb_tree_key_compareISt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2Ev,comdat
	.weak	_ZNSt20_Rb_tree_key_compareISt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2Ev # -- Begin function _ZNSt20_Rb_tree_key_compareISt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2Ev
	.p2align	4, 0x90
	.type	_ZNSt20_Rb_tree_key_compareISt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2Ev,@function
_ZNSt20_Rb_tree_key_compareISt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2Ev: # @_ZNSt20_Rb_tree_key_compareISt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2Ev
.Lfunc_begin31:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	popq	%rbp
	retq
.Lfunc_end31:
	.size	_ZNSt20_Rb_tree_key_compareISt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2Ev, .Lfunc_end31-_ZNSt20_Rb_tree_key_compareISt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt15_Rb_tree_headerC2Ev,"axG",@progbits,_ZNSt15_Rb_tree_headerC2Ev,comdat
	.weak	_ZNSt15_Rb_tree_headerC2Ev # -- Begin function _ZNSt15_Rb_tree_headerC2Ev
	.p2align	4, 0x90
	.type	_ZNSt15_Rb_tree_headerC2Ev,@function
_ZNSt15_Rb_tree_headerC2Ev:             # @_ZNSt15_Rb_tree_headerC2Ev
.Lfunc_begin32:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	$0, (%rdi)
	popq	%rbp
	jmp	_ZNSt15_Rb_tree_header8_M_resetEv # TAILCALL
.Lfunc_end32:
	.size	_ZNSt15_Rb_tree_headerC2Ev, .Lfunc_end32-_ZNSt15_Rb_tree_headerC2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt15_Rb_tree_header8_M_resetEv,"axG",@progbits,_ZNSt15_Rb_tree_header8_M_resetEv,comdat
	.weak	_ZNSt15_Rb_tree_header8_M_resetEv # -- Begin function _ZNSt15_Rb_tree_header8_M_resetEv
	.p2align	4, 0x90
	.type	_ZNSt15_Rb_tree_header8_M_resetEv,@function
_ZNSt15_Rb_tree_header8_M_resetEv:      # @_ZNSt15_Rb_tree_header8_M_resetEv
.Lfunc_begin33:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	$0, 8(%rdi)
	movq	%rdi, 16(%rdi)
	movq	%rdi, 24(%rdi)
	movq	$0, 32(%rdi)
	popq	%rbp
	retq
.Lfunc_end33:
	.size	_ZNSt15_Rb_tree_header8_M_resetEv, .Lfunc_end33-_ZNSt15_Rb_tree_header8_M_resetEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEC2Ev,"axG",@progbits,_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEC2Ev,comdat
	.weak	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEC2Ev # -- Begin function _ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEC2Ev
	.p2align	4, 0x90
	.type	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEC2Ev,@function
_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEC2Ev: # @_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEC2Ev
.Lfunc_begin34:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	popq	%rbp
	retq
.Lfunc_end34:
	.size	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEC2Ev, .Lfunc_end34-_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEC2Ev
	.cfi_endproc
                                        # -- End function
	.text
	.globl	_Z12create_tracePKc     # -- Begin function _Z12create_tracePKc
	.p2align	4, 0x90
	.type	_Z12create_tracePKc,@function
_Z12create_tracePKc:                    # @_Z12create_tracePKc
.Lfunc_begin35:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception3
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	callq	_ZTW5trace
	cmpq	$0, (%rax)
	je	.LBB35_2
# %bb.1:
	movabsq	$.L.str.3.72, %rdi
	movabsq	$.L.str.4.73, %rsi
	movl	$10, %edx
	movabsq	$.L__PRETTY_FUNCTION__._Z12create_tracePKc, %rcx
	callq	__assert_fail
.LBB35_2:
	movl	$88, %eax
	movl	%eax, %edi
	callq	_Znwm
	movq	%rax, %rdi
.Ltmp33:
	movq	%rdi, -16(%rbp)         # 8-byte Spill
	movq	%rax, %rdi
	movq	-8(%rbp), %rsi          # 8-byte Reload
	callq	_ZN10trace_infoC2EPKc
.Ltmp34:
	jmp	.LBB35_3
.LBB35_3:
	callq	_ZTW5trace
	movq	-16(%rbp), %rcx         # 8-byte Reload
	movq	%rcx, (%rax)
	addq	$32, %rsp
	popq	%rbp
	retq
.LBB35_4:
.Ltmp35:
	movl	%edx, %ecx
	movq	-16(%rbp), %rdi         # 8-byte Reload
	movl	%ecx, -20(%rbp)         # 4-byte Spill
	movq	%rax, -32(%rbp)         # 8-byte Spill
	callq	_ZdlPv
	movq	-32(%rbp), %rdi         # 8-byte Reload
	callq	_Unwind_Resume
.Lfunc_end35:
	.size	_Z12create_tracePKc, .Lfunc_end35-_Z12create_tracePKc
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2
GCC_except_table35:
.Lexception3:
	.byte	255                     # @LPStart Encoding = omit
	.byte	3                       # @TType Encoding = udata4
	.byte	41                      # @TType base offset
	.byte	3                       # Call site Encoding = udata4
	.byte	39                      # Call site table length
	.long	.Lfunc_begin35-.Lfunc_begin35 # >> Call Site 1 <<
	.long	.Ltmp33-.Lfunc_begin35  #   Call between .Lfunc_begin35 and .Ltmp33
	.long	0                       #     has no landing pad
	.byte	0                       #   On action: cleanup
	.long	.Ltmp33-.Lfunc_begin35  # >> Call Site 2 <<
	.long	.Ltmp34-.Ltmp33         #   Call between .Ltmp33 and .Ltmp34
	.long	.Ltmp35-.Lfunc_begin35  #     jumps to .Ltmp35
	.byte	0                       #   On action: cleanup
	.long	.Ltmp34-.Lfunc_begin35  # >> Call Site 3 <<
	.long	.Lfunc_end35-.Ltmp34    #   Call between .Ltmp34 and .Lfunc_end35
	.long	0                       #     has no landing pad
	.byte	0                       #   On action: cleanup
	.p2align	2
                                        # -- End function
	.text
	.hidden	_ZTW5trace              # -- Begin function _ZTW5trace
	.weak	_ZTW5trace
	.p2align	4, 0x90
	.type	_ZTW5trace,@function
_ZTW5trace:                             # @_ZTW5trace
.Lfunc_begin36:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%fs:0, %rax
	leaq	trace@TPOFF(%rax), %rax
	popq	%rbp
	retq
.Lfunc_end36:
	.size	_ZTW5trace, .Lfunc_end36-_ZTW5trace
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN10trace_infoC2EPKc,"axG",@progbits,_ZN10trace_infoC2EPKc,comdat
	.weak	_ZN10trace_infoC2EPKc   # -- Begin function _ZN10trace_infoC2EPKc
	.p2align	4, 0x90
	.type	_ZN10trace_infoC2EPKc,@function
_ZN10trace_infoC2EPKc:                  # @_ZN10trace_infoC2EPKc
.Lfunc_begin37:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception4
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	leaq	-8(%rbp), %rax
	movq	%rdi, -16(%rbp)         # 8-byte Spill
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	movq	%rsi, -32(%rbp)         # 8-byte Spill
	callq	_ZNSaIcEC2Ev
.Ltmp36:
	movq	-16(%rbp), %rdi         # 8-byte Reload
	movq	-32(%rbp), %rsi         # 8-byte Reload
	movq	-24(%rbp), %rdx         # 8-byte Reload
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS3_
.Ltmp37:
	jmp	.LBB37_1
.LBB37_1:
	leaq	-8(%rbp), %rdi
	callq	_ZNSaIcED2Ev
	movq	-16(%rbp), %rdi         # 8-byte Reload
	movq	$0, 40(%rdi)
	addq	$48, %rdi
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2Ev
	movq	-16(%rbp), %rdi         # 8-byte Reload
	movl	$1, 80(%rdi)
	addq	$48, %rsp
	popq	%rbp
	retq
.LBB37_2:
.Ltmp38:
	leaq	-8(%rbp), %rdi
	movl	%edx, %ecx
	movq	%rax, -40(%rbp)         # 8-byte Spill
	movl	%ecx, -44(%rbp)         # 4-byte Spill
	callq	_ZNSaIcED2Ev
	movq	-40(%rbp), %rdi         # 8-byte Reload
	callq	_Unwind_Resume
.Lfunc_end37:
	.size	_ZN10trace_infoC2EPKc, .Lfunc_end37-_ZN10trace_infoC2EPKc
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2
GCC_except_table37:
.Lexception4:
	.byte	255                     # @LPStart Encoding = omit
	.byte	3                       # @TType Encoding = udata4
	.asciz	"\234"                  # @TType base offset
	.byte	3                       # Call site Encoding = udata4
	.byte	26                      # Call site table length
	.long	.Ltmp36-.Lfunc_begin37  # >> Call Site 1 <<
	.long	.Ltmp37-.Ltmp36         #   Call between .Ltmp36 and .Ltmp37
	.long	.Ltmp38-.Lfunc_begin37  #     jumps to .Ltmp38
	.byte	0                       #   On action: cleanup
	.long	.Ltmp37-.Lfunc_begin37  # >> Call Site 2 <<
	.long	.Lfunc_end37-.Ltmp37    #   Call between .Ltmp37 and .Lfunc_end37
	.long	0                       #     has no landing pad
	.byte	0                       #   On action: cleanup
	.p2align	2
                                        # -- End function
	.text
	.globl	_Z14write_labelmapv     # -- Begin function _Z14write_labelmapv
	.p2align	4, 0x90
	.type	_Z14write_labelmapv,@function
_Z14write_labelmapv:                    # @_Z14write_labelmapv
.Lfunc_begin38:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	callq	_ZTW5trace
	movq	(%rax), %rax
	movq	32(%rax), %rax
	movl	$.L.str.5.74, %ecx
	movl	%ecx, %esi
	movl	$26, %edx
	movq	%rax, %rdi
	movq	%rax, -8(%rbp)          # 8-byte Spill
	callq	gzwrite
	movl	$_Z12labelmap_strB5cxx11, %ecx
	movl	%ecx, %esi
	movq	%rsi, %rdi
	movl	%eax, -12(%rbp)         # 4-byte Spill
	movq	%rsi, -24(%rbp)         # 8-byte Spill
	callq	_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5c_strEv
	movq	-24(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	callq	_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv
	movl	%eax, %ecx
	movq	-8(%rbp), %rdi          # 8-byte Reload
	movq	-32(%rbp), %rsi         # 8-byte Reload
	movl	%ecx, %edx
	callq	gzwrite
	movl	$.L.str.6.75, %ecx
	movl	%ecx, %esi
	movl	$25, %edx
	movq	-8(%rbp), %rdi          # 8-byte Reload
	movl	%eax, -36(%rbp)         # 4-byte Spill
	addq	$48, %rsp
	popq	%rbp
	jmp	gzwrite                 # TAILCALL
.Lfunc_end38:
	.size	_Z14write_labelmapv, .Lfunc_end38-_Z14write_labelmapv
	.cfi_endproc
                                        # -- End function
	.globl	_Z15open_trace_filev    # -- Begin function _Z15open_trace_filev
	.p2align	4, 0x90
	.type	_Z15open_trace_filev,@function
_Z15open_trace_filev:                   # @_Z15open_trace_filev
.Lfunc_begin39:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movabsq	$lock, %rdi
	callq	pthread_mutex_lock
	movl	%eax, -20(%rbp)         # 4-byte Spill
	callq	_ZTW5trace
	movabsq	$_Z8gz_filesB5cxx11, %rdi
	movq	(%rax), %rsi
	callq	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE4findERSB_
	movabsq	$_Z8gz_filesB5cxx11, %rdi
	movq	%rax, -8(%rbp)
	callq	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE3endEv
	leaq	-8(%rbp), %rdi
	leaq	-16(%rbp), %rsi
	movq	%rax, -16(%rbp)
	callq	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEneERKSB_
	movb	%al, -21(%rbp)          # 1-byte Spill
	callq	_ZTW5trace
	movq	(%rax), %rax
	movb	-21(%rbp), %cl          # 1-byte Reload
	testb	$1, %cl
	movq	%rax, -32(%rbp)         # 8-byte Spill
	jne	.LBB39_1
	jmp	.LBB39_2
.LBB39_1:
	movabsq	$_Z8gz_filesB5cxx11, %rdi
	movq	-32(%rbp), %rsi         # 8-byte Reload
	callq	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE2atERSB_
	movq	(%rax), %rax
	movq	%rax, -40(%rbp)         # 8-byte Spill
	callq	_ZTW5trace
	movq	(%rax), %rax
	movq	-40(%rbp), %rsi         # 8-byte Reload
	movq	%rsi, 32(%rax)
	jmp	.LBB39_5
.LBB39_2:
	movq	-32(%rbp), %rdi         # 8-byte Reload
	callq	_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5c_strEv
	movabsq	$.L.str.7.76, %rsi
	movq	%rax, %rdi
	callq	gzopen
	cmpq	$0, %rax
	movq	%rax, -48(%rbp)         # 8-byte Spill
	jne	.LBB39_4
# %bb.3:
	movabsq	$.L.str.8.77, %rdi
	callq	perror
	movl	$4294967295, %edi       # imm = 0xFFFFFFFF
	callq	exit
.LBB39_4:
	callq	_ZTW5trace
	movabsq	$_Z8gz_filesB5cxx11, %rdi
	movq	(%rax), %rsi
	callq	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEEixERSB_
	movq	-48(%rbp), %rsi         # 8-byte Reload
	movq	%rsi, (%rax)
	callq	_ZTW5trace
	movq	(%rax), %rax
	movq	-48(%rbp), %rsi         # 8-byte Reload
	movq	%rsi, 32(%rax)
	callq	_Z14write_labelmapv
.LBB39_5:
	movabsq	$lock, %rdi
	callq	pthread_mutex_unlock
	movl	%eax, -52(%rbp)         # 4-byte Spill
	addq	$64, %rsp
	popq	%rbp
	retq
.Lfunc_end39:
	.size	_Z15open_trace_filev, .Lfunc_end39-_Z15open_trace_filev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE4findERSB_,"axG",@progbits,_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE4findERSB_,comdat
	.weak	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE4findERSB_ # -- Begin function _ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE4findERSB_
	.p2align	4, 0x90
	.type	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE4findERSB_,@function
_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE4findERSB_: # @_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE4findERSB_
.Lfunc_begin40:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	jmp	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE4findERS7_ # TAILCALL
.Lfunc_end40:
	.size	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE4findERSB_, .Lfunc_end40-_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE4findERSB_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE3endEv,"axG",@progbits,_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE3endEv,comdat
	.weak	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE3endEv # -- Begin function _ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE3endEv
	.p2align	4, 0x90
	.type	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE3endEv,@function
_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE3endEv: # @_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE3endEv
.Lfunc_begin41:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	jmp	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE3endEv # TAILCALL
.Lfunc_end41:
	.size	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE3endEv, .Lfunc_end41-_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE3endEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEneERKSB_,"axG",@progbits,_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEneERKSB_,comdat
	.weak	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEneERKSB_ # -- Begin function _ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEneERKSB_
	.p2align	4, 0x90
	.type	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEneERKSB_,@function
_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEneERKSB_: # @_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEneERKSB_
.Lfunc_begin42:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	(%rdi), %rdi
	cmpq	(%rsi), %rdi
	setne	%al
	andb	$1, %al
	movzbl	%al, %eax
	popq	%rbp
	retq
.Lfunc_end42:
	.size	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEneERKSB_, .Lfunc_end42-_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEneERKSB_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE2atERSB_,"axG",@progbits,_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE2atERSB_,comdat
	.weak	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE2atERSB_ # -- Begin function _ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE2atERSB_
	.p2align	4, 0x90
	.type	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE2atERSB_,@function
_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE2atERSB_: # @_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE2atERSB_
.Lfunc_begin43:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	leaq	-8(%rbp), %rax
	movq	%rdi, -32(%rbp)         # 8-byte Spill
	movq	%rsi, -40(%rbp)         # 8-byte Spill
	movq	%rax, -48(%rbp)         # 8-byte Spill
	callq	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE11lower_boundERSB_
	leaq	-16(%rbp), %rsi
	movq	%rax, -8(%rbp)
	movq	-32(%rbp), %rdi         # 8-byte Reload
	movq	%rsi, -56(%rbp)         # 8-byte Spill
	callq	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE3endEv
	leaq	-8(%rbp), %rdi
	leaq	-16(%rbp), %rsi
	movq	%rax, -16(%rbp)
	callq	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEeqERKSB_
	testb	$1, %al
	jne	.LBB43_2
# %bb.1:
	movq	-32(%rbp), %rdi         # 8-byte Reload
	callq	_ZNKSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE8key_compEv
	leaq	-8(%rbp), %rdi
	callq	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEdeEv
	leaq	-24(%rbp), %rdi
	movq	-40(%rbp), %rsi         # 8-byte Reload
	movq	%rax, %rdx
	callq	_ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_
	testb	$1, %al
	jne	.LBB43_3
	jmp	.LBB43_4
.LBB43_2:
.LBB43_3:
	movabsq	$.L.str.36.78, %rdi
	callq	_ZSt20__throw_out_of_rangePKc
.LBB43_4:
	leaq	-8(%rbp), %rdi
	callq	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEdeEv
	addq	$32, %rax
	addq	$64, %rsp
	popq	%rbp
	retq
.Lfunc_end43:
	.size	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE2atERSB_, .Lfunc_end43-_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE2atERSB_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEEixERSB_,"axG",@progbits,_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEEixERSB_,comdat
	.weak	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEEixERSB_ # -- Begin function _ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEEixERSB_
	.p2align	4, 0x90
	.type	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEEixERSB_,@function
_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEEixERSB_: # @_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEEixERSB_
.Lfunc_begin44:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$96, %rsp
	leaq	-8(%rbp), %rax
	movq	%rdi, -56(%rbp)         # 8-byte Spill
	movq	%rsi, -64(%rbp)         # 8-byte Spill
	movq	%rax, -72(%rbp)         # 8-byte Spill
	callq	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE11lower_boundERSB_
	leaq	-16(%rbp), %rsi
	leaq	-8(%rbp), %rdi
	movq	%rax, -8(%rbp)
	movq	-56(%rbp), %rax         # 8-byte Reload
	movq	%rdi, -80(%rbp)         # 8-byte Spill
	movq	%rax, %rdi
	movq	%rsi, -88(%rbp)         # 8-byte Spill
	callq	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE3endEv
	leaq	-8(%rbp), %rdi
	leaq	-16(%rbp), %rsi
	movq	%rax, -16(%rbp)
	callq	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEeqERKSB_
	testb	$1, %al
	jne	.LBB44_2
# %bb.1:
	movq	-56(%rbp), %rdi         # 8-byte Reload
	callq	_ZNKSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE8key_compEv
	leaq	-8(%rbp), %rdi
	callq	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEdeEv
	leaq	-24(%rbp), %rdi
	movq	-64(%rbp), %rsi         # 8-byte Reload
	movq	%rax, %rdx
	callq	_ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_
	testb	$1, %al
	jne	.LBB44_3
	jmp	.LBB44_4
.LBB44_2:
.LBB44_3:
	leaq	-32(%rbp), %rdi
	leaq	-8(%rbp), %rsi
	callq	_ZNSt23_Rb_tree_const_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2ERKSt17_Rb_tree_iteratorISA_E
	leaq	-40(%rbp), %rdi
	movq	-64(%rbp), %rsi         # 8-byte Reload
	callq	_ZNSt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2IvLb1EEES7_
	movabsq	$_ZStL19piecewise_construct, %rdx
	leaq	-40(%rbp), %rcx
	leaq	-48(%rbp), %r8
	movq	-32(%rbp), %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE22_M_emplace_hint_uniqueIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEESt17_Rb_tree_iteratorISA_ESt23_Rb_tree_const_iteratorISA_EDpOT_
	movq	-80(%rbp), %rcx         # 8-byte Reload
	movq	%rax, (%rcx)
.LBB44_4:
	leaq	-8(%rbp), %rdi
	callq	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEdeEv
	addq	$32, %rax
	addq	$96, %rsp
	popq	%rbp
	retq
.Lfunc_end44:
	.size	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEEixERSB_, .Lfunc_end44-_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEEixERSB_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE11lower_boundERSB_,"axG",@progbits,_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE11lower_boundERSB_,comdat
	.weak	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE11lower_boundERSB_ # -- Begin function _ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE11lower_boundERSB_
	.p2align	4, 0x90
	.type	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE11lower_boundERSB_,@function
_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE11lower_boundERSB_: # @_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE11lower_boundERSB_
.Lfunc_begin45:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	jmp	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11lower_boundERS7_ # TAILCALL
.Lfunc_end45:
	.size	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE11lower_boundERSB_, .Lfunc_end45-_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE11lower_boundERSB_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEeqERKSB_,"axG",@progbits,_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEeqERKSB_,comdat
	.weak	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEeqERKSB_ # -- Begin function _ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEeqERKSB_
	.p2align	4, 0x90
	.type	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEeqERKSB_,@function
_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEeqERKSB_: # @_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEeqERKSB_
.Lfunc_begin46:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	(%rdi), %rdi
	cmpq	(%rsi), %rdi
	sete	%al
	andb	$1, %al
	movzbl	%al, %eax
	popq	%rbp
	retq
.Lfunc_end46:
	.size	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEeqERKSB_, .Lfunc_end46-_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEeqERKSB_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNKSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE8key_compEv,"axG",@progbits,_ZNKSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE8key_compEv,comdat
	.weak	_ZNKSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE8key_compEv # -- Begin function _ZNKSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE8key_compEv
	.p2align	4, 0x90
	.type	_ZNKSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE8key_compEv,@function
_ZNKSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE8key_compEv: # @_ZNKSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE8key_compEv
.Lfunc_begin47:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	jmp	_ZNKSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8key_compEv # TAILCALL
.Lfunc_end47:
	.size	_ZNKSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE8key_compEv, .Lfunc_end47-_ZNKSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE8key_compEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEdeEv,"axG",@progbits,_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEdeEv,comdat
	.weak	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEdeEv # -- Begin function _ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEdeEv
	.p2align	4, 0x90
	.type	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEdeEv,@function
_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEdeEv: # @_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEdeEv
.Lfunc_begin48:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	(%rdi), %rdi
	callq	_ZNSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv
	popq	%rbp
	retq
.Lfunc_end48:
	.size	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEdeEv, .Lfunc_end48-_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEdeEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_,"axG",@progbits,_ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_,comdat
	.weak	_ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_ # -- Begin function _ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_
	.p2align	4, 0x90
	.type	_ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_,@function
_ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_: # @_ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_
.Lfunc_begin49:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	movq	%rsi, %rdi
	movq	%rdx, %rsi
	popq	%rbp
	jmp	_ZStltIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EESA_ # TAILCALL
.Lfunc_end49:
	.size	_ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_, .Lfunc_end49-_ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt23_Rb_tree_const_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2ERKSt17_Rb_tree_iteratorISA_E,"axG",@progbits,_ZNSt23_Rb_tree_const_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2ERKSt17_Rb_tree_iteratorISA_E,comdat
	.weak	_ZNSt23_Rb_tree_const_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2ERKSt17_Rb_tree_iteratorISA_E # -- Begin function _ZNSt23_Rb_tree_const_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2ERKSt17_Rb_tree_iteratorISA_E
	.p2align	4, 0x90
	.type	_ZNSt23_Rb_tree_const_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2ERKSt17_Rb_tree_iteratorISA_E,@function
_ZNSt23_Rb_tree_const_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2ERKSt17_Rb_tree_iteratorISA_E: # @_ZNSt23_Rb_tree_const_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2ERKSt17_Rb_tree_iteratorISA_E
.Lfunc_begin50:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	(%rsi), %rsi
	movq	%rsi, (%rdi)
	popq	%rbp
	retq
.Lfunc_end50:
	.size	_ZNSt23_Rb_tree_const_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2ERKSt17_Rb_tree_iteratorISA_E, .Lfunc_end50-_ZNSt23_Rb_tree_const_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2ERKSt17_Rb_tree_iteratorISA_E
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2IvLb1EEES7_,"axG",@progbits,_ZNSt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2IvLb1EEES7_,comdat
	.weak	_ZNSt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2IvLb1EEES7_ # -- Begin function _ZNSt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2IvLb1EEES7_
	.p2align	4, 0x90
	.type	_ZNSt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2IvLb1EEES7_,@function
_ZNSt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2IvLb1EEES7_: # @_ZNSt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2IvLb1EEES7_
.Lfunc_begin51:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	jmp	_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2ES7_ # TAILCALL
.Lfunc_end51:
	.size	_ZNSt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2IvLb1EEES7_, .Lfunc_end51-_ZNSt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2IvLb1EEES7_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE22_M_emplace_hint_uniqueIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEESt17_Rb_tree_iteratorISA_ESt23_Rb_tree_const_iteratorISA_EDpOT_,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE22_M_emplace_hint_uniqueIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEESt17_Rb_tree_iteratorISA_ESt23_Rb_tree_const_iteratorISA_EDpOT_,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE22_M_emplace_hint_uniqueIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEESt17_Rb_tree_iteratorISA_ESt23_Rb_tree_const_iteratorISA_EDpOT_ # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE22_M_emplace_hint_uniqueIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEESt17_Rb_tree_iteratorISA_ESt23_Rb_tree_const_iteratorISA_EDpOT_
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE22_M_emplace_hint_uniqueIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEESt17_Rb_tree_iteratorISA_ESt23_Rb_tree_const_iteratorISA_EDpOT_,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE22_M_emplace_hint_uniqueIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEESt17_Rb_tree_iteratorISA_ESt23_Rb_tree_const_iteratorISA_EDpOT_: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE22_M_emplace_hint_uniqueIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEESt17_Rb_tree_iteratorISA_ESt23_Rb_tree_const_iteratorISA_EDpOT_
.Lfunc_begin52:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception5
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$144, %rsp
	movq	%rdi, -16(%rbp)         # 8-byte Spill
	movq	%rdx, %rdi
	movq	%rcx, -24(%rbp)         # 8-byte Spill
	movq	%r8, -32(%rbp)          # 8-byte Spill
	movq	%rsi, -40(%rbp)         # 8-byte Spill
	callq	_ZSt7forwardIRKSt21piecewise_construct_tEOT_RNSt16remove_referenceIS3_E4typeE
	movq	-24(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -48(%rbp)         # 8-byte Spill
	callq	_ZSt7forwardISt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEEOT_RNSt16remove_referenceISA_E4typeE
	movq	-32(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -56(%rbp)         # 8-byte Spill
	callq	_ZSt7forwardISt5tupleIJEEEOT_RNSt16remove_referenceIS2_E4typeE
	movq	-16(%rbp), %rdi         # 8-byte Reload
	movq	-48(%rbp), %rsi         # 8-byte Reload
	movq	-56(%rbp), %rdx         # 8-byte Reload
	movq	%rax, %rcx
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_create_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEPSt13_Rb_tree_nodeISA_EDpOT_
	movq	%rax, %rcx
.Ltmp39:
	movq	%rax, %rdi
	movq	%rcx, -64(%rbp)         # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt13_Rb_tree_nodeISA_E
.Ltmp40:
	movq	%rax, -72(%rbp)         # 8-byte Spill
	jmp	.LBB52_1
.LBB52_1:
.Ltmp41:
	movq	-16(%rbp), %rdi         # 8-byte Reload
	movq	-40(%rbp), %rsi         # 8-byte Reload
	movq	-72(%rbp), %rdx         # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorISA_ERS7_
.Ltmp42:
	movq	%rdx, -80(%rbp)         # 8-byte Spill
	movq	%rax, -88(%rbp)         # 8-byte Spill
	jmp	.LBB52_2
.LBB52_2:
	movq	-80(%rbp), %rax         # 8-byte Reload
	cmpq	$0, %rax
	je	.LBB52_6
# %bb.3:
.Ltmp43:
	movq	-16(%rbp), %rdi         # 8-byte Reload
	movq	-88(%rbp), %rsi         # 8-byte Reload
	movq	-80(%rbp), %rdx         # 8-byte Reload
	movq	-64(%rbp), %rcx         # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_insert_nodeEPSt18_Rb_tree_node_baseSI_PSt13_Rb_tree_nodeISA_E
.Ltmp44:
	movq	%rax, -96(%rbp)         # 8-byte Spill
	jmp	.LBB52_4
.LBB52_4:
	movq	-96(%rbp), %rax         # 8-byte Reload
	movq	%rax, -8(%rbp)
	movq	%rax, -104(%rbp)        # 8-byte Spill
	jmp	.LBB52_7
.LBB52_5:
.Ltmp45:
	movq	%rax, %rdi
	movq	%rdx, -112(%rbp)        # 8-byte Spill
	callq	__cxa_begin_catch
	movq	-16(%rbp), %rdi         # 8-byte Reload
	movq	-64(%rbp), %rsi         # 8-byte Reload
	movq	%rax, -120(%rbp)        # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_drop_nodeEPSt13_Rb_tree_nodeISA_E
.Ltmp46:
	callq	__cxa_rethrow
.Ltmp47:
	jmp	.LBB52_11
.LBB52_6:
	movq	-16(%rbp), %rdi         # 8-byte Reload
	movq	-64(%rbp), %rsi         # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_drop_nodeEPSt13_Rb_tree_nodeISA_E
	leaq	-8(%rbp), %rdi
	movq	-88(%rbp), %rsi         # 8-byte Reload
	callq	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2EPSt18_Rb_tree_node_base
	movq	-8(%rbp), %rsi
	movq	%rsi, -104(%rbp)        # 8-byte Spill
.LBB52_7:
	movq	-104(%rbp), %rax        # 8-byte Reload
	addq	$144, %rsp
	popq	%rbp
	retq
.LBB52_8:
.Ltmp48:
	movl	%edx, %ecx
.Ltmp49:
	movl	%ecx, -124(%rbp)        # 4-byte Spill
	movq	%rax, -136(%rbp)        # 8-byte Spill
	callq	__cxa_end_catch
.Ltmp50:
	jmp	.LBB52_9
.LBB52_9:
	movq	-136(%rbp), %rdi        # 8-byte Reload
	callq	_Unwind_Resume
.LBB52_10:
.Ltmp51:
	movl	%edx, %ecx
	movq	%rax, %rdi
	movl	%ecx, -140(%rbp)        # 4-byte Spill
	callq	__clang_call_terminate
.LBB52_11:
.Lfunc_end52:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE22_M_emplace_hint_uniqueIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEESt17_Rb_tree_iteratorISA_ESt23_Rb_tree_const_iteratorISA_EDpOT_, .Lfunc_end52-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE22_M_emplace_hint_uniqueIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEESt17_Rb_tree_iteratorISA_ESt23_Rb_tree_const_iteratorISA_EDpOT_
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2
GCC_except_table52:
.Lexception5:
	.byte	255                     # @LPStart Encoding = omit
	.byte	3                       # @TType Encoding = udata4
	.asciz	"\326\200\200"          # @TType base offset
	.byte	3                       # Call site Encoding = udata4
	.byte	78                      # Call site table length
	.long	.Lfunc_begin52-.Lfunc_begin52 # >> Call Site 1 <<
	.long	.Ltmp39-.Lfunc_begin52  #   Call between .Lfunc_begin52 and .Ltmp39
	.long	0                       #     has no landing pad
	.byte	0                       #   On action: cleanup
	.long	.Ltmp39-.Lfunc_begin52  # >> Call Site 2 <<
	.long	.Ltmp44-.Ltmp39         #   Call between .Ltmp39 and .Ltmp44
	.long	.Ltmp45-.Lfunc_begin52  #     jumps to .Ltmp45
	.byte	1                       #   On action: 1
	.long	.Ltmp44-.Lfunc_begin52  # >> Call Site 3 <<
	.long	.Ltmp46-.Ltmp44         #   Call between .Ltmp44 and .Ltmp46
	.long	0                       #     has no landing pad
	.byte	0                       #   On action: cleanup
	.long	.Ltmp46-.Lfunc_begin52  # >> Call Site 4 <<
	.long	.Ltmp47-.Ltmp46         #   Call between .Ltmp46 and .Ltmp47
	.long	.Ltmp48-.Lfunc_begin52  #     jumps to .Ltmp48
	.byte	0                       #   On action: cleanup
	.long	.Ltmp49-.Lfunc_begin52  # >> Call Site 5 <<
	.long	.Ltmp50-.Ltmp49         #   Call between .Ltmp49 and .Ltmp50
	.long	.Ltmp51-.Lfunc_begin52  #     jumps to .Ltmp51
	.byte	1                       #   On action: 1
	.long	.Ltmp50-.Lfunc_begin52  # >> Call Site 6 <<
	.long	.Lfunc_end52-.Ltmp50    #   Call between .Ltmp50 and .Lfunc_end52
	.long	0                       #     has no landing pad
	.byte	0                       #   On action: cleanup
	.byte	1                       # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                       #   No further actions
                                        # >> Catch TypeInfos <<
	.long	0                       # TypeInfo 1
	.p2align	2
                                        # -- End function
	.section	.text._ZSt7forwardIRKSt21piecewise_construct_tEOT_RNSt16remove_referenceIS3_E4typeE,"axG",@progbits,_ZSt7forwardIRKSt21piecewise_construct_tEOT_RNSt16remove_referenceIS3_E4typeE,comdat
	.weak	_ZSt7forwardIRKSt21piecewise_construct_tEOT_RNSt16remove_referenceIS3_E4typeE # -- Begin function _ZSt7forwardIRKSt21piecewise_construct_tEOT_RNSt16remove_referenceIS3_E4typeE
	.p2align	4, 0x90
	.type	_ZSt7forwardIRKSt21piecewise_construct_tEOT_RNSt16remove_referenceIS3_E4typeE,@function
_ZSt7forwardIRKSt21piecewise_construct_tEOT_RNSt16remove_referenceIS3_E4typeE: # @_ZSt7forwardIRKSt21piecewise_construct_tEOT_RNSt16remove_referenceIS3_E4typeE
.Lfunc_begin53:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, %rax
	popq	%rbp
	retq
.Lfunc_end53:
	.size	_ZSt7forwardIRKSt21piecewise_construct_tEOT_RNSt16remove_referenceIS3_E4typeE, .Lfunc_end53-_ZSt7forwardIRKSt21piecewise_construct_tEOT_RNSt16remove_referenceIS3_E4typeE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt7forwardISt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEEOT_RNSt16remove_referenceISA_E4typeE,"axG",@progbits,_ZSt7forwardISt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEEOT_RNSt16remove_referenceISA_E4typeE,comdat
	.weak	_ZSt7forwardISt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEEOT_RNSt16remove_referenceISA_E4typeE # -- Begin function _ZSt7forwardISt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEEOT_RNSt16remove_referenceISA_E4typeE
	.p2align	4, 0x90
	.type	_ZSt7forwardISt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEEOT_RNSt16remove_referenceISA_E4typeE,@function
_ZSt7forwardISt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEEOT_RNSt16remove_referenceISA_E4typeE: # @_ZSt7forwardISt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEEOT_RNSt16remove_referenceISA_E4typeE
.Lfunc_begin54:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, %rax
	popq	%rbp
	retq
.Lfunc_end54:
	.size	_ZSt7forwardISt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEEOT_RNSt16remove_referenceISA_E4typeE, .Lfunc_end54-_ZSt7forwardISt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEEOT_RNSt16remove_referenceISA_E4typeE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt7forwardISt5tupleIJEEEOT_RNSt16remove_referenceIS2_E4typeE,"axG",@progbits,_ZSt7forwardISt5tupleIJEEEOT_RNSt16remove_referenceIS2_E4typeE,comdat
	.weak	_ZSt7forwardISt5tupleIJEEEOT_RNSt16remove_referenceIS2_E4typeE # -- Begin function _ZSt7forwardISt5tupleIJEEEOT_RNSt16remove_referenceIS2_E4typeE
	.p2align	4, 0x90
	.type	_ZSt7forwardISt5tupleIJEEEOT_RNSt16remove_referenceIS2_E4typeE,@function
_ZSt7forwardISt5tupleIJEEEOT_RNSt16remove_referenceIS2_E4typeE: # @_ZSt7forwardISt5tupleIJEEEOT_RNSt16remove_referenceIS2_E4typeE
.Lfunc_begin55:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, %rax
	popq	%rbp
	retq
.Lfunc_end55:
	.size	_ZSt7forwardISt5tupleIJEEEOT_RNSt16remove_referenceIS2_E4typeE, .Lfunc_end55-_ZSt7forwardISt5tupleIJEEEOT_RNSt16remove_referenceIS2_E4typeE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_create_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEPSt13_Rb_tree_nodeISA_EDpOT_,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_create_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEPSt13_Rb_tree_nodeISA_EDpOT_,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_create_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEPSt13_Rb_tree_nodeISA_EDpOT_ # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_create_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEPSt13_Rb_tree_nodeISA_EDpOT_
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_create_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEPSt13_Rb_tree_nodeISA_EDpOT_,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_create_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEPSt13_Rb_tree_nodeISA_EDpOT_: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_create_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEPSt13_Rb_tree_nodeISA_EDpOT_
.Lfunc_begin56:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	movq	%rdx, -16(%rbp)         # 8-byte Spill
	movq	%rsi, -24(%rbp)         # 8-byte Spill
	movq	%rcx, -32(%rbp)         # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_get_nodeEv
	movq	-24(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -40(%rbp)         # 8-byte Spill
	callq	_ZSt7forwardIRKSt21piecewise_construct_tEOT_RNSt16remove_referenceIS3_E4typeE
	movq	-16(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -48(%rbp)         # 8-byte Spill
	callq	_ZSt7forwardISt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEEOT_RNSt16remove_referenceISA_E4typeE
	movq	-32(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -56(%rbp)         # 8-byte Spill
	callq	_ZSt7forwardISt5tupleIJEEEOT_RNSt16remove_referenceIS2_E4typeE
	movq	-8(%rbp), %rdi          # 8-byte Reload
	movq	-40(%rbp), %rsi         # 8-byte Reload
	movq	-48(%rbp), %rdx         # 8-byte Reload
	movq	-56(%rbp), %rcx         # 8-byte Reload
	movq	%rax, %r8
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE17_M_construct_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEvPSt13_Rb_tree_nodeISA_EDpOT_
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	retq
.Lfunc_end56:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_create_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEPSt13_Rb_tree_nodeISA_EDpOT_, .Lfunc_end56-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_create_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEPSt13_Rb_tree_nodeISA_EDpOT_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt13_Rb_tree_nodeISA_E,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt13_Rb_tree_nodeISA_E,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt13_Rb_tree_nodeISA_E # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt13_Rb_tree_nodeISA_E
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt13_Rb_tree_nodeISA_E,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt13_Rb_tree_nodeISA_E: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt13_Rb_tree_nodeISA_E
.Lfunc_begin57:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_valueEPKSt13_Rb_tree_nodeISA_E
	leaq	-8(%rbp), %rdi
	movq	%rax, %rsi
	callq	_ZNKSt10_Select1stISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEclERKSA_
	addq	$16, %rsp
	popq	%rbp
	retq
.Lfunc_end57:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt13_Rb_tree_nodeISA_E, .Lfunc_end57-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt13_Rb_tree_nodeISA_E
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorISA_ERS7_,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorISA_ERS7_,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorISA_ERS7_ # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorISA_ERS7_
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorISA_ERS7_,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorISA_ERS7_: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorISA_ERS7_
.Lfunc_begin58:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$176, %rsp
	leaq	-24(%rbp), %rax
	leaq	-32(%rbp), %rcx
	movq	%rsi, -24(%rbp)
	movq	%rdi, -96(%rbp)         # 8-byte Spill
	movq	%rax, %rdi
	movq	%rcx, -104(%rbp)        # 8-byte Spill
	movq	%rdx, -112(%rbp)        # 8-byte Spill
	callq	_ZNKSt23_Rb_tree_const_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE13_M_const_castEv
	leaq	-32(%rbp), %rcx
	movq	%rax, -32(%rbp)
	movq	-96(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -120(%rbp)        # 8-byte Spill
	movq	%rcx, -128(%rbp)        # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_M_endEv
	movq	-120(%rbp), %rcx        # 8-byte Reload
	cmpq	%rax, %rcx
	jne	.LBB58_5
# %bb.1:
	movq	-96(%rbp), %rdi         # 8-byte Reload
	callq	_ZNKSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE4sizeEv
	cmpq	$0, %rax
	je	.LBB58_4
# %bb.2:
	movq	-96(%rbp), %rdi         # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_rightmostEv
	movq	(%rax), %rdi
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt18_Rb_tree_node_base
	movq	-96(%rbp), %rdi         # 8-byte Reload
	movq	%rax, %rsi
	movq	-112(%rbp), %rdx        # 8-byte Reload
	callq	_ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_
	testb	$1, %al
	jne	.LBB58_3
	jmp	.LBB58_4
.LBB58_3:
	movq	$0, -40(%rbp)
	movq	-96(%rbp), %rdi         # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_rightmostEv
	leaq	-16(%rbp), %rdi
	leaq	-40(%rbp), %rsi
	movq	%rax, %rdx
	callq	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEERKS1_OT_
	jmp	.LBB58_24
.LBB58_4:
	movq	-96(%rbp), %rdi         # 8-byte Reload
	movq	-112(%rbp), %rsi        # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE24_M_get_insert_unique_posERS7_
	movq	%rax, -16(%rbp)
	movq	%rdx, -8(%rbp)
	jmp	.LBB58_24
.LBB58_5:
	movq	-120(%rbp), %rdi        # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt18_Rb_tree_node_base
	movq	-96(%rbp), %rdi         # 8-byte Reload
	movq	-112(%rbp), %rsi        # 8-byte Reload
	movq	%rax, %rdx
	callq	_ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_
	testb	$1, %al
	jne	.LBB58_6
	jmp	.LBB58_14
.LBB58_6:
	leaq	-48(%rbp), %rax
	movq	-120(%rbp), %rcx        # 8-byte Reload
	movq	%rcx, -48(%rbp)
	movq	-96(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -136(%rbp)        # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_leftmostEv
	movq	-120(%rbp), %rcx        # 8-byte Reload
	cmpq	%rcx, (%rax)
	jne	.LBB58_8
# %bb.7:
	movq	-96(%rbp), %rdi         # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_leftmostEv
	movq	-96(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -144(%rbp)        # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_leftmostEv
	leaq	-16(%rbp), %rdi
	movq	-144(%rbp), %rsi        # 8-byte Reload
	movq	%rax, %rdx
	callq	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_S4_Lb1EEEOT_OT0_
	jmp	.LBB58_13
.LBB58_8:
	leaq	-48(%rbp), %rdi
	callq	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEmmEv
	movq	(%rax), %rdi
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt18_Rb_tree_node_base
	movq	-96(%rbp), %rdi         # 8-byte Reload
	movq	%rax, %rsi
	movq	-112(%rbp), %rdx        # 8-byte Reload
	callq	_ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_
	testb	$1, %al
	jne	.LBB58_9
	jmp	.LBB58_12
.LBB58_9:
	leaq	-48(%rbp), %rdx
	movq	-48(%rbp), %rdi
	movq	%rdx, -152(%rbp)        # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_rightEPSt18_Rb_tree_node_base
	cmpq	$0, %rax
	jne	.LBB58_11
# %bb.10:
	leaq	-16(%rbp), %rdi
	leaq	-56(%rbp), %rsi
	movq	$0, -56(%rbp)
	movq	-152(%rbp), %rdx        # 8-byte Reload
	callq	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEERKS1_OT_
	jmp	.LBB58_13
.LBB58_11:
	leaq	-16(%rbp), %rdi
	movq	-128(%rbp), %rsi        # 8-byte Reload
	movq	-128(%rbp), %rdx        # 8-byte Reload
	callq	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_S4_Lb1EEEOT_OT0_
	jmp	.LBB58_13
.LBB58_12:
	movq	-96(%rbp), %rdi         # 8-byte Reload
	movq	-112(%rbp), %rsi        # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE24_M_get_insert_unique_posERS7_
	movq	%rax, -16(%rbp)
	movq	%rdx, -8(%rbp)
.LBB58_13:
	jmp	.LBB58_24
.LBB58_14:
	movq	-120(%rbp), %rdi        # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt18_Rb_tree_node_base
	movq	-96(%rbp), %rdi         # 8-byte Reload
	movq	%rax, %rsi
	movq	-112(%rbp), %rdx        # 8-byte Reload
	callq	_ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_
	testb	$1, %al
	jne	.LBB58_15
	jmp	.LBB58_23
.LBB58_15:
	leaq	-64(%rbp), %rax
	movq	-32(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	movq	-96(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -160(%rbp)        # 8-byte Spill
	movq	%rcx, -168(%rbp)        # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_rightmostEv
	movq	-168(%rbp), %rcx        # 8-byte Reload
	cmpq	%rcx, (%rax)
	jne	.LBB58_17
# %bb.16:
	movq	$0, -72(%rbp)
	movq	-96(%rbp), %rdi         # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_rightmostEv
	leaq	-16(%rbp), %rdi
	leaq	-72(%rbp), %rsi
	movq	%rax, %rdx
	callq	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEERKS1_OT_
	jmp	.LBB58_22
.LBB58_17:
	leaq	-64(%rbp), %rdi
	callq	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEppEv
	movq	(%rax), %rdi
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt18_Rb_tree_node_base
	movq	-96(%rbp), %rdi         # 8-byte Reload
	movq	-112(%rbp), %rsi        # 8-byte Reload
	movq	%rax, %rdx
	callq	_ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_
	testb	$1, %al
	jne	.LBB58_18
	jmp	.LBB58_21
.LBB58_18:
	movq	-128(%rbp), %rax        # 8-byte Reload
	movq	(%rax), %rdi
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_rightEPSt18_Rb_tree_node_base
	cmpq	$0, %rax
	jne	.LBB58_20
# %bb.19:
	leaq	-16(%rbp), %rdi
	leaq	-80(%rbp), %rsi
	movq	$0, -80(%rbp)
	movq	-128(%rbp), %rdx        # 8-byte Reload
	callq	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEERKS1_OT_
	jmp	.LBB58_22
.LBB58_20:
	leaq	-16(%rbp), %rdi
	leaq	-64(%rbp), %rax
	movq	%rax, %rsi
	movq	%rax, %rdx
	callq	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_S4_Lb1EEEOT_OT0_
	jmp	.LBB58_22
.LBB58_21:
	movq	-96(%rbp), %rdi         # 8-byte Reload
	movq	-112(%rbp), %rsi        # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE24_M_get_insert_unique_posERS7_
	movq	%rax, -16(%rbp)
	movq	%rdx, -8(%rbp)
.LBB58_22:
	jmp	.LBB58_24
.LBB58_23:
	leaq	-16(%rbp), %rdi
	leaq	-88(%rbp), %rdx
	movq	$0, -88(%rbp)
	movq	-128(%rbp), %rsi        # 8-byte Reload
	callq	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEEOT_RKS1_
.LBB58_24:
	movq	-16(%rbp), %rax
	movq	-8(%rbp), %rdx
	addq	$176, %rsp
	popq	%rbp
	retq
.Lfunc_end58:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorISA_ERS7_, .Lfunc_end58-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorISA_ERS7_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_insert_nodeEPSt18_Rb_tree_node_baseSI_PSt13_Rb_tree_nodeISA_E,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_insert_nodeEPSt18_Rb_tree_node_baseSI_PSt13_Rb_tree_nodeISA_E,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_insert_nodeEPSt18_Rb_tree_node_baseSI_PSt13_Rb_tree_nodeISA_E # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_insert_nodeEPSt18_Rb_tree_node_baseSI_PSt13_Rb_tree_nodeISA_E
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_insert_nodeEPSt18_Rb_tree_node_baseSI_PSt13_Rb_tree_nodeISA_E,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_insert_nodeEPSt18_Rb_tree_node_baseSI_PSt13_Rb_tree_nodeISA_E: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_insert_nodeEPSt18_Rb_tree_node_baseSI_PSt13_Rb_tree_nodeISA_E
.Lfunc_begin59:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movb	$1, %al
	cmpq	$0, %rsi
	movq	%rdx, -16(%rbp)         # 8-byte Spill
	movq	%rcx, -24(%rbp)         # 8-byte Spill
	movq	%rdi, -32(%rbp)         # 8-byte Spill
	movb	%al, -33(%rbp)          # 1-byte Spill
	jne	.LBB59_3
# %bb.1:
	movq	-32(%rbp), %rdi         # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_M_endEv
	movb	$1, %cl
	movq	-16(%rbp), %rdi         # 8-byte Reload
	cmpq	%rdi, %rax
	movb	%cl, -33(%rbp)          # 1-byte Spill
	je	.LBB59_3
# %bb.2:
	movq	-24(%rbp), %rdi         # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt13_Rb_tree_nodeISA_E
	movq	-16(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -48(%rbp)         # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt18_Rb_tree_node_base
	movq	-32(%rbp), %rdi         # 8-byte Reload
	movq	-48(%rbp), %rsi         # 8-byte Reload
	movq	%rax, %rdx
	callq	_ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_
	movb	%al, -33(%rbp)          # 1-byte Spill
.LBB59_3:
	movb	-33(%rbp), %al          # 1-byte Reload
	movq	-32(%rbp), %rcx         # 8-byte Reload
	addq	$8, %rcx
	andb	$1, %al
	movzbl	%al, %edi
	movq	-24(%rbp), %rsi         # 8-byte Reload
	movq	-16(%rbp), %rdx         # 8-byte Reload
	callq	_ZSt29_Rb_tree_insert_and_rebalancebPSt18_Rb_tree_node_baseS0_RS_
	leaq	-8(%rbp), %rdi
	movq	-32(%rbp), %rcx         # 8-byte Reload
	movq	40(%rcx), %rdx
	addq	$1, %rdx
	movq	%rdx, 40(%rcx)
	movq	-24(%rbp), %rsi         # 8-byte Reload
	callq	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2EPSt18_Rb_tree_node_base
	movq	-8(%rbp), %rax
	addq	$48, %rsp
	popq	%rbp
	retq
.Lfunc_end59:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_insert_nodeEPSt18_Rb_tree_node_baseSI_PSt13_Rb_tree_nodeISA_E, .Lfunc_end59-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_insert_nodeEPSt18_Rb_tree_node_baseSI_PSt13_Rb_tree_nodeISA_E
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2EPSt18_Rb_tree_node_base,"axG",@progbits,_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2EPSt18_Rb_tree_node_base,comdat
	.weak	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2EPSt18_Rb_tree_node_base # -- Begin function _ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2EPSt18_Rb_tree_node_base
	.p2align	4, 0x90
	.type	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2EPSt18_Rb_tree_node_base,@function
_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2EPSt18_Rb_tree_node_base: # @_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2EPSt18_Rb_tree_node_base
.Lfunc_begin60:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rsi, (%rdi)
	popq	%rbp
	retq
.Lfunc_end60:
	.size	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2EPSt18_Rb_tree_node_base, .Lfunc_end60-_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2EPSt18_Rb_tree_node_base
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_M_endEv,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_M_endEv,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_M_endEv # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_M_endEv
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_M_endEv,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_M_endEv: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_M_endEv
.Lfunc_begin61:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	addq	$8, %rdi
	movq	%rdi, %rax
	popq	%rbp
	retq
.Lfunc_end61:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_M_endEv, .Lfunc_end61-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_M_endEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt18_Rb_tree_node_base,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt18_Rb_tree_node_base,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt18_Rb_tree_node_base # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt18_Rb_tree_node_base
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt18_Rb_tree_node_base,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt18_Rb_tree_node_base: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt18_Rb_tree_node_base
.Lfunc_begin62:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_valueEPKSt18_Rb_tree_node_base
	leaq	-8(%rbp), %rdi
	movq	%rax, %rsi
	callq	_ZNKSt10_Select1stISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEclERKSA_
	addq	$16, %rsp
	popq	%rbp
	retq
.Lfunc_end62:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt18_Rb_tree_node_base, .Lfunc_end62-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt18_Rb_tree_node_base
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_valueEPKSt18_Rb_tree_node_base,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_valueEPKSt18_Rb_tree_node_base,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_valueEPKSt18_Rb_tree_node_base # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_valueEPKSt18_Rb_tree_node_base
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_valueEPKSt18_Rb_tree_node_base,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_valueEPKSt18_Rb_tree_node_base: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_valueEPKSt18_Rb_tree_node_base
.Lfunc_begin63:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	callq	_ZNKSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv
	popq	%rbp
	retq
.Lfunc_end63:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_valueEPKSt18_Rb_tree_node_base, .Lfunc_end63-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_valueEPKSt18_Rb_tree_node_base
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNKSt10_Select1stISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEclERKSA_,"axG",@progbits,_ZNKSt10_Select1stISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEclERKSA_,comdat
	.weak	_ZNKSt10_Select1stISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEclERKSA_ # -- Begin function _ZNKSt10_Select1stISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEclERKSA_
	.p2align	4, 0x90
	.type	_ZNKSt10_Select1stISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEclERKSA_,@function
_ZNKSt10_Select1stISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEclERKSA_: # @_ZNKSt10_Select1stISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEclERKSA_
.Lfunc_begin64:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rsi, %rax
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	popq	%rbp
	retq
.Lfunc_end64:
	.size	_ZNKSt10_Select1stISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEclERKSA_, .Lfunc_end64-_ZNKSt10_Select1stISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEclERKSA_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNKSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv,"axG",@progbits,_ZNKSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv,comdat
	.weak	_ZNKSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv # -- Begin function _ZNKSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv
	.p2align	4, 0x90
	.type	_ZNKSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv,@function
_ZNKSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv: # @_ZNKSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv
.Lfunc_begin65:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	addq	$32, %rdi
	popq	%rbp
	jmp	_ZNK9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE6_M_ptrEv # TAILCALL
.Lfunc_end65:
	.size	_ZNKSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv, .Lfunc_end65-_ZNKSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE6_M_ptrEv,"axG",@progbits,_ZNK9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE6_M_ptrEv,comdat
	.weak	_ZNK9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE6_M_ptrEv # -- Begin function _ZNK9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE6_M_ptrEv
	.p2align	4, 0x90
	.type	_ZNK9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE6_M_ptrEv,@function
_ZNK9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE6_M_ptrEv: # @_ZNK9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE6_M_ptrEv
.Lfunc_begin66:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	jmp	_ZNK9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE7_M_addrEv # TAILCALL
.Lfunc_end66:
	.size	_ZNK9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE6_M_ptrEv, .Lfunc_end66-_ZNK9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE6_M_ptrEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE7_M_addrEv,"axG",@progbits,_ZNK9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE7_M_addrEv,comdat
	.weak	_ZNK9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE7_M_addrEv # -- Begin function _ZNK9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE7_M_addrEv
	.p2align	4, 0x90
	.type	_ZNK9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE7_M_addrEv,@function
_ZNK9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE7_M_addrEv: # @_ZNK9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE7_M_addrEv
.Lfunc_begin67:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, %rax
	popq	%rbp
	retq
.Lfunc_end67:
	.size	_ZNK9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE7_M_addrEv, .Lfunc_end67-_ZNK9__gnu_cxx16__aligned_membufISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE7_M_addrEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNKSt23_Rb_tree_const_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE13_M_const_castEv,"axG",@progbits,_ZNKSt23_Rb_tree_const_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE13_M_const_castEv,comdat
	.weak	_ZNKSt23_Rb_tree_const_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE13_M_const_castEv # -- Begin function _ZNKSt23_Rb_tree_const_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE13_M_const_castEv
	.p2align	4, 0x90
	.type	_ZNKSt23_Rb_tree_const_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE13_M_const_castEv,@function
_ZNKSt23_Rb_tree_const_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE13_M_const_castEv: # @_ZNKSt23_Rb_tree_const_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE13_M_const_castEv
.Lfunc_begin68:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	leaq	-8(%rbp), %rax
	movq	(%rdi), %rsi
	movq	%rax, %rdi
	callq	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2EPSt18_Rb_tree_node_base
	movq	-8(%rbp), %rax
	addq	$16, %rsp
	popq	%rbp
	retq
.Lfunc_end68:
	.size	_ZNKSt23_Rb_tree_const_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE13_M_const_castEv, .Lfunc_end68-_ZNKSt23_Rb_tree_const_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE13_M_const_castEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNKSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE4sizeEv,"axG",@progbits,_ZNKSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE4sizeEv,comdat
	.weak	_ZNKSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE4sizeEv # -- Begin function _ZNKSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE4sizeEv
	.p2align	4, 0x90
	.type	_ZNKSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE4sizeEv,@function
_ZNKSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE4sizeEv: # @_ZNKSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE4sizeEv
.Lfunc_begin69:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	40(%rdi), %rax
	popq	%rbp
	retq
.Lfunc_end69:
	.size	_ZNKSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE4sizeEv, .Lfunc_end69-_ZNKSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE4sizeEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_rightmostEv,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_rightmostEv,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_rightmostEv # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_rightmostEv
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_rightmostEv,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_rightmostEv: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_rightmostEv
.Lfunc_begin70:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	addq	$32, %rdi
	movq	%rdi, %rax
	popq	%rbp
	retq
.Lfunc_end70:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_rightmostEv, .Lfunc_end70-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE12_M_rightmostEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEERKS1_OT_,"axG",@progbits,_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEERKS1_OT_,comdat
	.weak	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEERKS1_OT_ # -- Begin function _ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEERKS1_OT_
	.p2align	4, 0x90
	.type	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEERKS1_OT_,@function
_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEERKS1_OT_: # @_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEERKS1_OT_
.Lfunc_begin71:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	(%rsi), %rsi
	movq	%rsi, (%rdi)
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	movq	%rdx, %rdi
	callq	_ZSt7forwardIRPSt18_Rb_tree_node_baseEOT_RNSt16remove_referenceIS3_E4typeE
	movq	(%rax), %rax
	movq	-8(%rbp), %rdx          # 8-byte Reload
	movq	%rax, 8(%rdx)
	addq	$16, %rsp
	popq	%rbp
	retq
.Lfunc_end71:
	.size	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEERKS1_OT_, .Lfunc_end71-_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEERKS1_OT_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE24_M_get_insert_unique_posERS7_,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE24_M_get_insert_unique_posERS7_,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE24_M_get_insert_unique_posERS7_ # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE24_M_get_insert_unique_posERS7_
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE24_M_get_insert_unique_posERS7_,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE24_M_get_insert_unique_posERS7_: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE24_M_get_insert_unique_posERS7_
.Lfunc_begin72:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$224, %rsp
	leaq	-24(%rbp), %rax
	movq	%rdi, -64(%rbp)         # 8-byte Spill
	movq	%rax, -72(%rbp)         # 8-byte Spill
	movq	%rsi, -80(%rbp)         # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_beginEv
	leaq	-32(%rbp), %rsi
	movq	%rax, -24(%rbp)
	movq	-64(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -88(%rbp)         # 8-byte Spill
	movq	%rsi, -96(%rbp)         # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_M_endEv
	movq	%rax, -32(%rbp)
	movq	-88(%rbp), %rsi         # 8-byte Reload
	cmpq	$0, %rsi
	movq	%rax, -104(%rbp)        # 8-byte Spill
	jne	.LBB72_2
# %bb.1:
	leaq	-40(%rbp), %rax
	movq	%rax, %rcx
	movq	%rax, %rdi
	movq	-104(%rbp), %rsi        # 8-byte Reload
	movq	%rcx, -112(%rbp)        # 8-byte Spill
	callq	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2EPSt18_Rb_tree_node_base
	movq	-112(%rbp), %rax        # 8-byte Reload
	movq	%rax, -120(%rbp)        # 8-byte Spill
	jmp	.LBB72_8
.LBB72_2:
	movq	-88(%rbp), %rax         # 8-byte Reload
	movq	%rax, -128(%rbp)        # 8-byte Spill
.LBB72_3:                               # =>This Inner Loop Header: Depth=1
	movq	-128(%rbp), %rax        # 8-byte Reload
	movq	%rax, %rdi
	movq	%rax, -136(%rbp)        # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt13_Rb_tree_nodeISA_E
	movq	-64(%rbp), %rdi         # 8-byte Reload
	movq	-80(%rbp), %rsi         # 8-byte Reload
	movq	%rax, %rdx
	callq	_ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_
	testb	$1, %al
	movb	%al, -137(%rbp)         # 1-byte Spill
	jne	.LBB72_4
	jmp	.LBB72_5
.LBB72_4:                               #   in Loop: Header=BB72_3 Depth=1
	movq	-136(%rbp), %rdi        # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE7_S_leftEPSt18_Rb_tree_node_base
	movq	%rax, -152(%rbp)        # 8-byte Spill
	jmp	.LBB72_6
.LBB72_5:                               #   in Loop: Header=BB72_3 Depth=1
	movq	-136(%rbp), %rdi        # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_rightEPSt18_Rb_tree_node_base
	movq	%rax, -152(%rbp)        # 8-byte Spill
.LBB72_6:                               #   in Loop: Header=BB72_3 Depth=1
	movq	-152(%rbp), %rax        # 8-byte Reload
	cmpq	$0, %rax
	movq	%rax, %rcx
	movq	%rax, -160(%rbp)        # 8-byte Spill
	movq	%rcx, -128(%rbp)        # 8-byte Spill
	jne	.LBB72_3
# %bb.7:
	leaq	-40(%rbp), %rax
	movq	-160(%rbp), %rcx        # 8-byte Reload
	movq	%rcx, -24(%rbp)
	movq	-136(%rbp), %rdx        # 8-byte Reload
	movq	%rdx, -32(%rbp)
	movq	%rax, %rsi
	movq	%rax, %rdi
	movq	%rsi, -168(%rbp)        # 8-byte Spill
	movq	%rdx, %rsi
	callq	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2EPSt18_Rb_tree_node_base
	movb	-137(%rbp), %r8b        # 1-byte Reload
	testb	$1, %r8b
	movq	-168(%rbp), %rax        # 8-byte Reload
	movq	-168(%rbp), %rcx        # 8-byte Reload
	movq	%rax, -120(%rbp)        # 8-byte Spill
	movq	%rcx, -176(%rbp)        # 8-byte Spill
	jne	.LBB72_8
	jmp	.LBB72_11
.LBB72_8:
	movq	-120(%rbp), %rax        # 8-byte Reload
	movq	-64(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -184(%rbp)        # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE5beginEv
	leaq	-40(%rbp), %rdi
	leaq	-48(%rbp), %rsi
	movq	%rax, -48(%rbp)
	callq	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEeqERKSB_
	testb	$1, %al
	jne	.LBB72_9
	jmp	.LBB72_10
.LBB72_9:
	leaq	-16(%rbp), %rdi
	leaq	-24(%rbp), %rsi
	leaq	-32(%rbp), %rdx
	callq	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRPSt13_Rb_tree_nodeIS_IKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEERS1_Lb1EEEOT_OT0_
	movq	-184(%rbp), %rdx        # 8-byte Reload
	movq	%rdx, -192(%rbp)        # 8-byte Spill
	jmp	.LBB72_14
.LBB72_10:
	leaq	-40(%rbp), %rdi
	callq	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEmmEv
	movq	-184(%rbp), %rdi        # 8-byte Reload
	movq	%rax, -200(%rbp)        # 8-byte Spill
	movq	%rdi, -176(%rbp)        # 8-byte Spill
.LBB72_11:
	movq	-176(%rbp), %rax        # 8-byte Reload
	leaq	-40(%rbp), %rsi
	movq	-40(%rbp), %rdi
	movq	%rax, -208(%rbp)        # 8-byte Spill
	movq	%rsi, -216(%rbp)        # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt18_Rb_tree_node_base
	movq	-64(%rbp), %rdi         # 8-byte Reload
	movq	%rax, %rsi
	movq	-80(%rbp), %rdx         # 8-byte Reload
	callq	_ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_
	testb	$1, %al
	jne	.LBB72_12
	jmp	.LBB72_13
.LBB72_12:
	leaq	-16(%rbp), %rdi
	leaq	-24(%rbp), %rsi
	leaq	-32(%rbp), %rdx
	callq	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRPSt13_Rb_tree_nodeIS_IKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEERS1_Lb1EEEOT_OT0_
	movq	-208(%rbp), %rdx        # 8-byte Reload
	movq	%rdx, -192(%rbp)        # 8-byte Spill
	jmp	.LBB72_14
.LBB72_13:
	leaq	-16(%rbp), %rdi
	leaq	-56(%rbp), %rdx
	movq	$0, -56(%rbp)
	movq	-216(%rbp), %rsi        # 8-byte Reload
	callq	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEEOT_RKS1_
	movq	-208(%rbp), %rdx        # 8-byte Reload
	movq	%rdx, -192(%rbp)        # 8-byte Spill
.LBB72_14:
	movq	-192(%rbp), %rax        # 8-byte Reload
	movq	-16(%rbp), %rcx
	movq	-8(%rbp), %rdx
	movq	%rax, -224(%rbp)        # 8-byte Spill
	movq	%rcx, %rax
	addq	$224, %rsp
	popq	%rbp
	retq
.Lfunc_end72:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE24_M_get_insert_unique_posERS7_, .Lfunc_end72-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE24_M_get_insert_unique_posERS7_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_leftmostEv,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_leftmostEv,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_leftmostEv # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_leftmostEv
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_leftmostEv,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_leftmostEv: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_leftmostEv
.Lfunc_begin73:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	addq	$24, %rdi
	movq	%rdi, %rax
	popq	%rbp
	retq
.Lfunc_end73:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_leftmostEv, .Lfunc_end73-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_leftmostEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_S4_Lb1EEEOT_OT0_,"axG",@progbits,_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_S4_Lb1EEEOT_OT0_,comdat
	.weak	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_S4_Lb1EEEOT_OT0_ # -- Begin function _ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_S4_Lb1EEEOT_OT0_
	.p2align	4, 0x90
	.type	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_S4_Lb1EEEOT_OT0_,@function
_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_S4_Lb1EEEOT_OT0_: # @_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_S4_Lb1EEEOT_OT0_
.Lfunc_begin74:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	movq	%rsi, %rdi
	movq	%rdx, -16(%rbp)         # 8-byte Spill
	callq	_ZSt7forwardIRPSt18_Rb_tree_node_baseEOT_RNSt16remove_referenceIS3_E4typeE
	movq	(%rax), %rax
	movq	-8(%rbp), %rdx          # 8-byte Reload
	movq	%rax, (%rdx)
	movq	-16(%rbp), %rdi         # 8-byte Reload
	callq	_ZSt7forwardIRPSt18_Rb_tree_node_baseEOT_RNSt16remove_referenceIS3_E4typeE
	movq	(%rax), %rax
	movq	-8(%rbp), %rdx          # 8-byte Reload
	movq	%rax, 8(%rdx)
	addq	$16, %rsp
	popq	%rbp
	retq
.Lfunc_end74:
	.size	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_S4_Lb1EEEOT_OT0_, .Lfunc_end74-_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_S4_Lb1EEEOT_OT0_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEmmEv,"axG",@progbits,_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEmmEv,comdat
	.weak	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEmmEv # -- Begin function _ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEmmEv
	.p2align	4, 0x90
	.type	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEmmEv,@function
_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEmmEv: # @_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEmmEv
.Lfunc_begin75:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	(%rdi), %rax
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	movq	%rax, %rdi
	callq	_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base
	movq	-8(%rbp), %rdi          # 8-byte Reload
	movq	%rax, (%rdi)
	movq	%rdi, %rax
	addq	$16, %rsp
	popq	%rbp
	retq
.Lfunc_end75:
	.size	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEmmEv, .Lfunc_end75-_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEmmEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEppEv,"axG",@progbits,_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEppEv,comdat
	.weak	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEppEv # -- Begin function _ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEppEv
	.p2align	4, 0x90
	.type	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEppEv,@function
_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEppEv: # @_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEppEv
.Lfunc_begin76:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	(%rdi), %rax
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	movq	%rax, %rdi
	callq	_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base
	movq	-8(%rbp), %rdi          # 8-byte Reload
	movq	%rax, (%rdi)
	movq	%rdi, %rax
	addq	$16, %rsp
	popq	%rbp
	retq
.Lfunc_end76:
	.size	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEppEv, .Lfunc_end76-_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEppEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEEOT_RKS1_,"axG",@progbits,_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEEOT_RKS1_,comdat
	.weak	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEEOT_RKS1_ # -- Begin function _ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEEOT_RKS1_
	.p2align	4, 0x90
	.type	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEEOT_RKS1_,@function
_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEEOT_RKS1_: # @_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEEOT_RKS1_
.Lfunc_begin77:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	movq	%rsi, %rdi
	movq	%rdx, -16(%rbp)         # 8-byte Spill
	callq	_ZSt7forwardIRPSt18_Rb_tree_node_baseEOT_RNSt16remove_referenceIS3_E4typeE
	movq	(%rax), %rax
	movq	-8(%rbp), %rdx          # 8-byte Reload
	movq	%rax, (%rdx)
	movq	-16(%rbp), %rax         # 8-byte Reload
	movq	(%rax), %rsi
	movq	%rsi, 8(%rdx)
	addq	$16, %rsp
	popq	%rbp
	retq
.Lfunc_end77:
	.size	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEEOT_RKS1_, .Lfunc_end77-_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRS1_Lb1EEEOT_RKS1_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt7forwardIRPSt18_Rb_tree_node_baseEOT_RNSt16remove_referenceIS3_E4typeE,"axG",@progbits,_ZSt7forwardIRPSt18_Rb_tree_node_baseEOT_RNSt16remove_referenceIS3_E4typeE,comdat
	.weak	_ZSt7forwardIRPSt18_Rb_tree_node_baseEOT_RNSt16remove_referenceIS3_E4typeE # -- Begin function _ZSt7forwardIRPSt18_Rb_tree_node_baseEOT_RNSt16remove_referenceIS3_E4typeE
	.p2align	4, 0x90
	.type	_ZSt7forwardIRPSt18_Rb_tree_node_baseEOT_RNSt16remove_referenceIS3_E4typeE,@function
_ZSt7forwardIRPSt18_Rb_tree_node_baseEOT_RNSt16remove_referenceIS3_E4typeE: # @_ZSt7forwardIRPSt18_Rb_tree_node_baseEOT_RNSt16remove_referenceIS3_E4typeE
.Lfunc_begin78:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, %rax
	popq	%rbp
	retq
.Lfunc_end78:
	.size	_ZSt7forwardIRPSt18_Rb_tree_node_baseEOT_RNSt16remove_referenceIS3_E4typeE, .Lfunc_end78-_ZSt7forwardIRPSt18_Rb_tree_node_baseEOT_RNSt16remove_referenceIS3_E4typeE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE5beginEv,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE5beginEv,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE5beginEv # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE5beginEv
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE5beginEv,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE5beginEv: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE5beginEv
.Lfunc_begin79:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	leaq	-8(%rbp), %rax
	movq	24(%rdi), %rsi
	movq	%rax, %rdi
	callq	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2EPSt18_Rb_tree_node_base
	movq	-8(%rbp), %rax
	addq	$16, %rsp
	popq	%rbp
	retq
.Lfunc_end79:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE5beginEv, .Lfunc_end79-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE5beginEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRPSt13_Rb_tree_nodeIS_IKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEERS1_Lb1EEEOT_OT0_,"axG",@progbits,_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRPSt13_Rb_tree_nodeIS_IKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEERS1_Lb1EEEOT_OT0_,comdat
	.weak	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRPSt13_Rb_tree_nodeIS_IKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEERS1_Lb1EEEOT_OT0_ # -- Begin function _ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRPSt13_Rb_tree_nodeIS_IKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEERS1_Lb1EEEOT_OT0_
	.p2align	4, 0x90
	.type	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRPSt13_Rb_tree_nodeIS_IKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEERS1_Lb1EEEOT_OT0_,@function
_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRPSt13_Rb_tree_nodeIS_IKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEERS1_Lb1EEEOT_OT0_: # @_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRPSt13_Rb_tree_nodeIS_IKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEERS1_Lb1EEEOT_OT0_
.Lfunc_begin80:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	movq	%rsi, %rdi
	movq	%rdx, -16(%rbp)         # 8-byte Spill
	callq	_ZSt7forwardIRPSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEOT_RNSt16remove_referenceISF_E4typeE
	movq	(%rax), %rax
	movq	-8(%rbp), %rdx          # 8-byte Reload
	movq	%rax, (%rdx)
	movq	-16(%rbp), %rdi         # 8-byte Reload
	callq	_ZSt7forwardIRPSt18_Rb_tree_node_baseEOT_RNSt16remove_referenceIS3_E4typeE
	movq	(%rax), %rax
	movq	-8(%rbp), %rdx          # 8-byte Reload
	movq	%rax, 8(%rdx)
	addq	$16, %rsp
	popq	%rbp
	retq
.Lfunc_end80:
	.size	_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRPSt13_Rb_tree_nodeIS_IKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEERS1_Lb1EEEOT_OT0_, .Lfunc_end80-_ZNSt4pairIPSt18_Rb_tree_node_baseS1_EC2IRPSt13_Rb_tree_nodeIS_IKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEERS1_Lb1EEEOT_OT0_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt7forwardIRPSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEOT_RNSt16remove_referenceISF_E4typeE,"axG",@progbits,_ZSt7forwardIRPSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEOT_RNSt16remove_referenceISF_E4typeE,comdat
	.weak	_ZSt7forwardIRPSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEOT_RNSt16remove_referenceISF_E4typeE # -- Begin function _ZSt7forwardIRPSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEOT_RNSt16remove_referenceISF_E4typeE
	.p2align	4, 0x90
	.type	_ZSt7forwardIRPSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEOT_RNSt16remove_referenceISF_E4typeE,@function
_ZSt7forwardIRPSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEOT_RNSt16remove_referenceISF_E4typeE: # @_ZSt7forwardIRPSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEOT_RNSt16remove_referenceISF_E4typeE
.Lfunc_begin81:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, %rax
	popq	%rbp
	retq
.Lfunc_end81:
	.size	_ZSt7forwardIRPSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEOT_RNSt16remove_referenceISF_E4typeE, .Lfunc_end81-_ZSt7forwardIRPSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEOT_RNSt16remove_referenceISF_E4typeE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_valueEPKSt13_Rb_tree_nodeISA_E,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_valueEPKSt13_Rb_tree_nodeISA_E,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_valueEPKSt13_Rb_tree_nodeISA_E # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_valueEPKSt13_Rb_tree_nodeISA_E
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_valueEPKSt13_Rb_tree_nodeISA_E,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_valueEPKSt13_Rb_tree_nodeISA_E: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_valueEPKSt13_Rb_tree_nodeISA_E
.Lfunc_begin82:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	callq	_ZNKSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv
	popq	%rbp
	retq
.Lfunc_end82:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_valueEPKSt13_Rb_tree_nodeISA_E, .Lfunc_end82-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_valueEPKSt13_Rb_tree_nodeISA_E
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_get_nodeEv,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_get_nodeEv,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_get_nodeEv # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_get_nodeEv
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_get_nodeEv,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_get_nodeEv: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_get_nodeEv
.Lfunc_begin83:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE21_M_get_Node_allocatorEv
	movl	$1, %ecx
	movl	%ecx, %esi
	movq	%rax, %rdi
	popq	%rbp
	jmp	_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE8allocateERSD_m # TAILCALL
.Lfunc_end83:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_get_nodeEv, .Lfunc_end83-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_get_nodeEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE17_M_construct_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEvPSt13_Rb_tree_nodeISA_EDpOT_,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE17_M_construct_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEvPSt13_Rb_tree_nodeISA_EDpOT_,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE17_M_construct_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEvPSt13_Rb_tree_nodeISA_EDpOT_ # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE17_M_construct_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEvPSt13_Rb_tree_nodeISA_EDpOT_
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE17_M_construct_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEvPSt13_Rb_tree_nodeISA_EDpOT_,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE17_M_construct_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEvPSt13_Rb_tree_nodeISA_EDpOT_: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE17_M_construct_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEvPSt13_Rb_tree_nodeISA_EDpOT_
.Lfunc_begin84:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception6
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	movq	%rcx, -16(%rbp)         # 8-byte Spill
	movq	%rdx, -24(%rbp)         # 8-byte Spill
	movq	%rsi, -32(%rbp)         # 8-byte Spill
	movq	%r8, -40(%rbp)          # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE21_M_get_Node_allocatorEv
	movq	-32(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -48(%rbp)         # 8-byte Spill
	callq	_ZNSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv
	movq	-24(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -56(%rbp)         # 8-byte Spill
	callq	_ZSt7forwardIRKSt21piecewise_construct_tEOT_RNSt16remove_referenceIS3_E4typeE
	movq	-16(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -64(%rbp)         # 8-byte Spill
	callq	_ZSt7forwardISt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEEOT_RNSt16remove_referenceISA_E4typeE
	movq	-40(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -72(%rbp)         # 8-byte Spill
	callq	_ZSt7forwardISt5tupleIJEEEOT_RNSt16remove_referenceIS2_E4typeE
.Ltmp52:
	movq	-48(%rbp), %rdi         # 8-byte Reload
	movq	-56(%rbp), %rsi         # 8-byte Reload
	movq	-64(%rbp), %rdx         # 8-byte Reload
	movq	-72(%rbp), %rcx         # 8-byte Reload
	movq	%rax, %r8
	callq	_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE9constructISB_JRKSt21piecewise_construct_tSt5tupleIJRS8_EESJ_IJEEEEEvRSD_PT_DpOT0_
.Ltmp53:
	jmp	.LBB84_3
.LBB84_1:
.Ltmp54:
	movq	%rax, %rdi
	movq	%rdx, -80(%rbp)         # 8-byte Spill
	callq	__cxa_begin_catch
	movq	-8(%rbp), %rdi          # 8-byte Reload
	movq	-32(%rbp), %rsi         # 8-byte Reload
	movq	%rax, -88(%rbp)         # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11_M_put_nodeEPSt13_Rb_tree_nodeISA_E
.Ltmp55:
	callq	__cxa_rethrow
.Ltmp56:
	jmp	.LBB84_6
.LBB84_2:
.Ltmp57:
	movl	%edx, %ecx
.Ltmp58:
	movl	%ecx, -92(%rbp)         # 4-byte Spill
	movq	%rax, -104(%rbp)        # 8-byte Spill
	callq	__cxa_end_catch
.Ltmp59:
	jmp	.LBB84_4
.LBB84_3:
	addq	$112, %rsp
	popq	%rbp
	retq
.LBB84_4:
	movq	-104(%rbp), %rdi        # 8-byte Reload
	callq	_Unwind_Resume
.LBB84_5:
.Ltmp60:
	movl	%edx, %ecx
	movq	%rax, %rdi
	movl	%ecx, -108(%rbp)        # 4-byte Spill
	callq	__clang_call_terminate
.LBB84_6:
.Lfunc_end84:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE17_M_construct_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEvPSt13_Rb_tree_nodeISA_EDpOT_, .Lfunc_end84-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE17_M_construct_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEvPSt13_Rb_tree_nodeISA_EDpOT_
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2
GCC_except_table84:
.Lexception6:
	.byte	255                     # @LPStart Encoding = omit
	.byte	3                       # @TType Encoding = udata4
	.byte	73                      # @TType base offset
	.byte	3                       # Call site Encoding = udata4
	.byte	65                      # Call site table length
	.long	.Ltmp52-.Lfunc_begin84  # >> Call Site 1 <<
	.long	.Ltmp53-.Ltmp52         #   Call between .Ltmp52 and .Ltmp53
	.long	.Ltmp54-.Lfunc_begin84  #     jumps to .Ltmp54
	.byte	1                       #   On action: 1
	.long	.Ltmp53-.Lfunc_begin84  # >> Call Site 2 <<
	.long	.Ltmp55-.Ltmp53         #   Call between .Ltmp53 and .Ltmp55
	.long	0                       #     has no landing pad
	.byte	0                       #   On action: cleanup
	.long	.Ltmp55-.Lfunc_begin84  # >> Call Site 3 <<
	.long	.Ltmp56-.Ltmp55         #   Call between .Ltmp55 and .Ltmp56
	.long	.Ltmp57-.Lfunc_begin84  #     jumps to .Ltmp57
	.byte	0                       #   On action: cleanup
	.long	.Ltmp58-.Lfunc_begin84  # >> Call Site 4 <<
	.long	.Ltmp59-.Ltmp58         #   Call between .Ltmp58 and .Ltmp59
	.long	.Ltmp60-.Lfunc_begin84  #     jumps to .Ltmp60
	.byte	1                       #   On action: 1
	.long	.Ltmp59-.Lfunc_begin84  # >> Call Site 5 <<
	.long	.Lfunc_end84-.Ltmp59    #   Call between .Ltmp59 and .Lfunc_end84
	.long	0                       #     has no landing pad
	.byte	0                       #   On action: cleanup
	.byte	1                       # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                       #   No further actions
                                        # >> Catch TypeInfos <<
	.long	0                       # TypeInfo 1
	.p2align	2
                                        # -- End function
	.section	.text._ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE9constructISB_JRKSt21piecewise_construct_tSt5tupleIJRS8_EESJ_IJEEEEEvRSD_PT_DpOT0_,"axG",@progbits,_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE9constructISB_JRKSt21piecewise_construct_tSt5tupleIJRS8_EESJ_IJEEEEEvRSD_PT_DpOT0_,comdat
	.weak	_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE9constructISB_JRKSt21piecewise_construct_tSt5tupleIJRS8_EESJ_IJEEEEEvRSD_PT_DpOT0_ # -- Begin function _ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE9constructISB_JRKSt21piecewise_construct_tSt5tupleIJRS8_EESJ_IJEEEEEvRSD_PT_DpOT0_
	.p2align	4, 0x90
	.type	_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE9constructISB_JRKSt21piecewise_construct_tSt5tupleIJRS8_EESJ_IJEEEEEvRSD_PT_DpOT0_,@function
_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE9constructISB_JRKSt21piecewise_construct_tSt5tupleIJRS8_EESJ_IJEEEEEvRSD_PT_DpOT0_: # @_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE9constructISB_JRKSt21piecewise_construct_tSt5tupleIJRS8_EESJ_IJEEEEEvRSD_PT_DpOT0_
.Lfunc_begin85:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	movq	%rdx, %rdi
	movq	%rcx, -16(%rbp)         # 8-byte Spill
	movq	%r8, -24(%rbp)          # 8-byte Spill
	movq	%rsi, -32(%rbp)         # 8-byte Spill
	callq	_ZSt7forwardIRKSt21piecewise_construct_tEOT_RNSt16remove_referenceIS3_E4typeE
	movq	-16(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -40(%rbp)         # 8-byte Spill
	callq	_ZSt7forwardISt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEEOT_RNSt16remove_referenceISA_E4typeE
	movq	-24(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -48(%rbp)         # 8-byte Spill
	callq	_ZSt7forwardISt5tupleIJEEEOT_RNSt16remove_referenceIS2_E4typeE
	movq	-8(%rbp), %rdi          # 8-byte Reload
	movq	-32(%rbp), %rsi         # 8-byte Reload
	movq	-40(%rbp), %rdx         # 8-byte Reload
	movq	-48(%rbp), %rcx         # 8-byte Reload
	movq	%rax, %r8
	addq	$48, %rsp
	popq	%rbp
	jmp	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE9constructISC_JRKSt21piecewise_construct_tSt5tupleIJRS9_EESJ_IJEEEEEvPT_DpOT0_ # TAILCALL
.Lfunc_end85:
	.size	_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE9constructISB_JRKSt21piecewise_construct_tSt5tupleIJRS8_EESJ_IJEEEEEvRSD_PT_DpOT0_, .Lfunc_end85-_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE9constructISB_JRKSt21piecewise_construct_tSt5tupleIJRS8_EESJ_IJEEEEEvRSD_PT_DpOT0_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE9constructISC_JRKSt21piecewise_construct_tSt5tupleIJRS9_EESJ_IJEEEEEvPT_DpOT0_,"axG",@progbits,_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE9constructISC_JRKSt21piecewise_construct_tSt5tupleIJRS9_EESJ_IJEEEEEvPT_DpOT0_,comdat
	.weak	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE9constructISC_JRKSt21piecewise_construct_tSt5tupleIJRS9_EESJ_IJEEEEEvPT_DpOT0_ # -- Begin function _ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE9constructISC_JRKSt21piecewise_construct_tSt5tupleIJRS9_EESJ_IJEEEEEvPT_DpOT0_
	.p2align	4, 0x90
	.type	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE9constructISC_JRKSt21piecewise_construct_tSt5tupleIJRS9_EESJ_IJEEEEEvPT_DpOT0_,@function
_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE9constructISC_JRKSt21piecewise_construct_tSt5tupleIJRS9_EESJ_IJEEEEEvPT_DpOT0_: # @_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE9constructISC_JRKSt21piecewise_construct_tSt5tupleIJRS9_EESJ_IJEEEEEvPT_DpOT0_
.Lfunc_begin86:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -16(%rbp)         # 8-byte Spill
	movq	%rdx, %rdi
	movq	%rcx, -24(%rbp)         # 8-byte Spill
	movq	%r8, -32(%rbp)          # 8-byte Spill
	movq	%rsi, -40(%rbp)         # 8-byte Spill
	callq	_ZSt7forwardIRKSt21piecewise_construct_tEOT_RNSt16remove_referenceIS3_E4typeE
	movq	-24(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -48(%rbp)         # 8-byte Spill
	callq	_ZSt7forwardISt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEEOT_RNSt16remove_referenceISA_E4typeE
	leaq	-8(%rbp), %rdi
	movq	%rax, %rsi
	callq	_ZNSt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2EOS8_
	movq	-32(%rbp), %rdi         # 8-byte Reload
	callq	_ZSt7forwardISt5tupleIJEEEOT_RNSt16remove_referenceIS2_E4typeE
	leaq	-8(%rbp), %rsi
	movq	-40(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -56(%rbp)         # 8-byte Spill
	callq	_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEC2IJRS6_EJEEESt21piecewise_construct_tSt5tupleIJDpT_EESD_IJDpT0_EE
	addq	$64, %rsp
	popq	%rbp
	retq
.Lfunc_end86:
	.size	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE9constructISC_JRKSt21piecewise_construct_tSt5tupleIJRS9_EESJ_IJEEEEEvPT_DpOT0_, .Lfunc_end86-_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE9constructISC_JRKSt21piecewise_construct_tSt5tupleIJRS9_EESJ_IJEEEEEvPT_DpOT0_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2EOS8_,"axG",@progbits,_ZNSt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2EOS8_,comdat
	.weak	_ZNSt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2EOS8_ # -- Begin function _ZNSt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2EOS8_
	.p2align	4, 0x90
	.type	_ZNSt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2EOS8_,@function
_ZNSt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2EOS8_: # @_ZNSt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2EOS8_
.Lfunc_begin87:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	jmp	_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2EOS8_ # TAILCALL
.Lfunc_end87:
	.size	_ZNSt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2EOS8_, .Lfunc_end87-_ZNSt5tupleIJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2EOS8_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEC2IJRS6_EJEEESt21piecewise_construct_tSt5tupleIJDpT_EESD_IJDpT0_EE,"axG",@progbits,_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEC2IJRS6_EJEEESt21piecewise_construct_tSt5tupleIJDpT_EESD_IJDpT0_EE,comdat
	.weak	_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEC2IJRS6_EJEEESt21piecewise_construct_tSt5tupleIJDpT_EESD_IJDpT0_EE # -- Begin function _ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEC2IJRS6_EJEEESt21piecewise_construct_tSt5tupleIJDpT_EESD_IJDpT0_EE
	.p2align	4, 0x90
	.type	_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEC2IJRS6_EJEEESt21piecewise_construct_tSt5tupleIJDpT_EESD_IJDpT0_EE,@function
_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEC2IJRS6_EJEEESt21piecewise_construct_tSt5tupleIJDpT_EESD_IJDpT0_EE: # @_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEC2IJRS6_EJEEESt21piecewise_construct_tSt5tupleIJDpT_EESD_IJDpT0_EE
.Lfunc_begin88:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	leaq	-8(%rbp), %rdx
	callq	_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEC2IJRS6_EJLm0EEJEJEEERSt5tupleIJDpT_EERSC_IJDpT1_EESt12_Index_tupleIJXspT0_EEESL_IJXspT2_EEE
	addq	$16, %rsp
	popq	%rbp
	retq
.Lfunc_end88:
	.size	_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEC2IJRS6_EJEEESt21piecewise_construct_tSt5tupleIJDpT_EESD_IJDpT0_EE, .Lfunc_end88-_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEC2IJRS6_EJEEESt21piecewise_construct_tSt5tupleIJDpT_EESD_IJDpT0_EE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEC2IJRS6_EJLm0EEJEJEEERSt5tupleIJDpT_EERSC_IJDpT1_EESt12_Index_tupleIJXspT0_EEESL_IJXspT2_EEE,"axG",@progbits,_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEC2IJRS6_EJLm0EEJEJEEERSt5tupleIJDpT_EERSC_IJDpT1_EESt12_Index_tupleIJXspT0_EEESL_IJXspT2_EEE,comdat
	.weak	_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEC2IJRS6_EJLm0EEJEJEEERSt5tupleIJDpT_EERSC_IJDpT1_EESt12_Index_tupleIJXspT0_EEESL_IJXspT2_EEE # -- Begin function _ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEC2IJRS6_EJLm0EEJEJEEERSt5tupleIJDpT_EERSC_IJDpT1_EESt12_Index_tupleIJXspT0_EEESL_IJXspT2_EEE
	.p2align	4, 0x90
	.type	_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEC2IJRS6_EJLm0EEJEJEEERSt5tupleIJDpT_EERSC_IJDpT1_EESt12_Index_tupleIJXspT0_EEESL_IJXspT2_EEE,@function
_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEC2IJRS6_EJLm0EEJEJEEERSt5tupleIJDpT_EERSC_IJDpT1_EESt12_Index_tupleIJXspT0_EEESL_IJXspT2_EEE: # @_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEC2IJRS6_EJLm0EEJEJEEERSt5tupleIJDpT_EERSC_IJDpT1_EESt12_Index_tupleIJXspT0_EEESL_IJXspT2_EEE
.Lfunc_begin89:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	movq	%rsi, %rdi
	movq	%rdx, -16(%rbp)         # 8-byte Spill
	callq	_ZSt3getILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERSC_
	movq	%rax, %rdi
	callq	_ZSt7forwardIRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEOT_RNSt16remove_referenceIS8_E4typeE
	movq	-8(%rbp), %rdi          # 8-byte Reload
	movq	%rax, %rsi
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2ERKS4_
	movq	-8(%rbp), %rax          # 8-byte Reload
	movq	$0, 32(%rax)
	addq	$16, %rsp
	popq	%rbp
	retq
.Lfunc_end89:
	.size	_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEC2IJRS6_EJLm0EEJEJEEERSt5tupleIJDpT_EERSC_IJDpT1_EESt12_Index_tupleIJXspT0_EEESL_IJXspT2_EEE, .Lfunc_end89-_ZNSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEC2IJRS6_EJLm0EEJEJEEERSt5tupleIJDpT_EERSC_IJDpT1_EESt12_Index_tupleIJXspT0_EEESL_IJXspT2_EEE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt3getILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERSC_,"axG",@progbits,_ZSt3getILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERSC_,comdat
	.weak	_ZSt3getILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERSC_ # -- Begin function _ZSt3getILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERSC_
	.p2align	4, 0x90
	.type	_ZSt3getILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERSC_,@function
_ZSt3getILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERSC_: # @_ZSt3getILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERSC_
.Lfunc_begin90:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	jmp	_ZSt12__get_helperILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEJEERT0_RSt11_Tuple_implIXT_EJS8_DpT1_EE # TAILCALL
.Lfunc_end90:
	.size	_ZSt3getILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERSC_, .Lfunc_end90-_ZSt3getILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERSC_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt7forwardIRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEOT_RNSt16remove_referenceIS8_E4typeE,"axG",@progbits,_ZSt7forwardIRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEOT_RNSt16remove_referenceIS8_E4typeE,comdat
	.weak	_ZSt7forwardIRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEOT_RNSt16remove_referenceIS8_E4typeE # -- Begin function _ZSt7forwardIRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEOT_RNSt16remove_referenceIS8_E4typeE
	.p2align	4, 0x90
	.type	_ZSt7forwardIRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEOT_RNSt16remove_referenceIS8_E4typeE,@function
_ZSt7forwardIRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEOT_RNSt16remove_referenceIS8_E4typeE: # @_ZSt7forwardIRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEOT_RNSt16remove_referenceIS8_E4typeE
.Lfunc_begin91:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, %rax
	popq	%rbp
	retq
.Lfunc_end91:
	.size	_ZSt7forwardIRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEOT_RNSt16remove_referenceIS8_E4typeE, .Lfunc_end91-_ZSt7forwardIRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEOT_RNSt16remove_referenceIS8_E4typeE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt12__get_helperILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEJEERT0_RSt11_Tuple_implIXT_EJS8_DpT1_EE,"axG",@progbits,_ZSt12__get_helperILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEJEERT0_RSt11_Tuple_implIXT_EJS8_DpT1_EE,comdat
	.weak	_ZSt12__get_helperILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEJEERT0_RSt11_Tuple_implIXT_EJS8_DpT1_EE # -- Begin function _ZSt12__get_helperILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEJEERT0_RSt11_Tuple_implIXT_EJS8_DpT1_EE
	.p2align	4, 0x90
	.type	_ZSt12__get_helperILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEJEERT0_RSt11_Tuple_implIXT_EJS8_DpT1_EE,@function
_ZSt12__get_helperILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEJEERT0_RSt11_Tuple_implIXT_EJS8_DpT1_EE: # @_ZSt12__get_helperILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEJEERT0_RSt11_Tuple_implIXT_EJS8_DpT1_EE
.Lfunc_begin92:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	jmp	_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE7_M_headERS8_ # TAILCALL
.Lfunc_end92:
	.size	_ZSt12__get_helperILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEJEERT0_RSt11_Tuple_implIXT_EJS8_DpT1_EE, .Lfunc_end92-_ZSt12__get_helperILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEJEERT0_RSt11_Tuple_implIXT_EJS8_DpT1_EE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE7_M_headERS8_,"axG",@progbits,_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE7_M_headERS8_,comdat
	.weak	_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE7_M_headERS8_ # -- Begin function _ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE7_M_headERS8_
	.p2align	4, 0x90
	.type	_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE7_M_headERS8_,@function
_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE7_M_headERS8_: # @_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE7_M_headERS8_
.Lfunc_begin93:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	jmp	_ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EE7_M_headERS8_ # TAILCALL
.Lfunc_end93:
	.size	_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE7_M_headERS8_, .Lfunc_end93-_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE7_M_headERS8_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EE7_M_headERS8_,"axG",@progbits,_ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EE7_M_headERS8_,comdat
	.weak	_ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EE7_M_headERS8_ # -- Begin function _ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EE7_M_headERS8_
	.p2align	4, 0x90
	.type	_ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EE7_M_headERS8_,@function
_ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EE7_M_headERS8_: # @_ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EE7_M_headERS8_
.Lfunc_begin94:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	(%rdi), %rax
	popq	%rbp
	retq
.Lfunc_end94:
	.size	_ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EE7_M_headERS8_, .Lfunc_end94-_ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EE7_M_headERS8_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2EOS8_,"axG",@progbits,_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2EOS8_,comdat
	.weak	_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2EOS8_ # -- Begin function _ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2EOS8_
	.p2align	4, 0x90
	.type	_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2EOS8_,@function
_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2EOS8_: # @_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2EOS8_
.Lfunc_begin95:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	movq	%rsi, %rdi
	callq	_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE7_M_headERS8_
	movq	%rax, %rdi
	callq	_ZSt7forwardIRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEOT_RNSt16remove_referenceIS8_E4typeE
	movq	-8(%rbp), %rdi          # 8-byte Reload
	movq	%rax, %rsi
	addq	$16, %rsp
	popq	%rbp
	jmp	_ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EEC2ES7_ # TAILCALL
.Lfunc_end95:
	.size	_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2EOS8_, .Lfunc_end95-_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2EOS8_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EEC2ES7_,"axG",@progbits,_ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EEC2ES7_,comdat
	.weak	_ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EEC2ES7_ # -- Begin function _ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EEC2ES7_
	.p2align	4, 0x90
	.type	_ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EEC2ES7_,@function
_ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EEC2ES7_: # @_ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EEC2ES7_
.Lfunc_begin96:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rsi, (%rdi)
	popq	%rbp
	retq
.Lfunc_end96:
	.size	_ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EEC2ES7_, .Lfunc_end96-_ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EEC2ES7_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE8allocateERSD_m,"axG",@progbits,_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE8allocateERSD_m,comdat
	.weak	_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE8allocateERSD_m # -- Begin function _ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE8allocateERSD_m
	.p2align	4, 0x90
	.type	_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE8allocateERSD_m,@function
_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE8allocateERSD_m: # @_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE8allocateERSD_m
.Lfunc_begin97:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	xorl	%eax, %eax
	movl	%eax, %edx
	popq	%rbp
	jmp	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE8allocateEmPKv # TAILCALL
.Lfunc_end97:
	.size	_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE8allocateERSD_m, .Lfunc_end97-_ZNSt16allocator_traitsISaISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEEE8allocateERSD_m
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE8allocateEmPKv,"axG",@progbits,_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE8allocateEmPKv,comdat
	.weak	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE8allocateEmPKv # -- Begin function _ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE8allocateEmPKv
	.p2align	4, 0x90
	.type	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE8allocateEmPKv,@function
_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE8allocateEmPKv: # @_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE8allocateEmPKv
.Lfunc_begin98:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rsi, -8(%rbp)          # 8-byte Spill
	movq	%rdx, -16(%rbp)         # 8-byte Spill
	callq	_ZNK9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE8max_sizeEv
	movq	-8(%rbp), %rdx          # 8-byte Reload
	cmpq	%rdx, %rax
	jae	.LBB98_2
# %bb.1:
	callq	_ZSt17__throw_bad_allocv
.LBB98_2:
	movq	-8(%rbp), %rax          # 8-byte Reload
	shlq	$3, %rax
	leaq	(%rax,%rax,8), %rdi
	addq	$16, %rsp
	popq	%rbp
	jmp	_Znwm                   # TAILCALL
.Lfunc_end98:
	.size	_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE8allocateEmPKv, .Lfunc_end98-_ZN9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE8allocateEmPKv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE8max_sizeEv,"axG",@progbits,_ZNK9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE8max_sizeEv,comdat
	.weak	_ZNK9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE8max_sizeEv # -- Begin function _ZNK9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE8max_sizeEv
	.p2align	4, 0x90
	.type	_ZNK9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE8max_sizeEv,@function
_ZNK9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE8max_sizeEv: # @_ZNK9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE8max_sizeEv
.Lfunc_begin99:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movabsq	$256204778801521550, %rax # imm = 0x38E38E38E38E38E
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	popq	%rbp
	retq
.Lfunc_end99:
	.size	_ZNK9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE8max_sizeEv, .Lfunc_end99-_ZNK9__gnu_cxx13new_allocatorISt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEE8max_sizeEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2ES7_,"axG",@progbits,_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2ES7_,comdat
	.weak	_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2ES7_ # -- Begin function _ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2ES7_
	.p2align	4, 0x90
	.type	_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2ES7_,@function
_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2ES7_: # @_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2ES7_
.Lfunc_begin100:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	jmp	_ZNSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EEC2ES7_ # TAILCALL
.Lfunc_end100:
	.size	_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2ES7_, .Lfunc_end100-_ZNSt11_Tuple_implILm0EJRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEC2ES7_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZStltIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EESA_,"axG",@progbits,_ZStltIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EESA_,comdat
	.weak	_ZStltIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EESA_ # -- Begin function _ZStltIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EESA_
	.p2align	4, 0x90
	.type	_ZStltIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EESA_,@function
_ZStltIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EESA_: # @_ZStltIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EESA_
.Lfunc_begin101:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception7
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
.Ltmp61:
	callq	_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7compareERKS4_
.Ltmp62:
	movl	%eax, -4(%rbp)          # 4-byte Spill
	jmp	.LBB101_1
.LBB101_1:
	movl	-4(%rbp), %eax          # 4-byte Reload
	cmpl	$0, %eax
	setl	%cl
	andb	$1, %cl
	movzbl	%cl, %eax
	addq	$16, %rsp
	popq	%rbp
	retq
.LBB101_2:
.Ltmp63:
	movl	%edx, %ecx
	movq	%rax, %rdi
	movl	%ecx, -8(%rbp)          # 4-byte Spill
	callq	__clang_call_terminate
.Lfunc_end101:
	.size	_ZStltIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EESA_, .Lfunc_end101-_ZStltIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EESA_
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2
GCC_except_table101:
.Lexception7:
	.byte	255                     # @LPStart Encoding = omit
	.byte	3                       # @TType Encoding = udata4
	.byte	21                      # @TType base offset
	.byte	3                       # Call site Encoding = udata4
	.byte	13                      # Call site table length
	.long	.Ltmp61-.Lfunc_begin101 # >> Call Site 1 <<
	.long	.Ltmp62-.Ltmp61         #   Call between .Ltmp61 and .Ltmp62
	.long	.Ltmp63-.Lfunc_begin101 #     jumps to .Ltmp63
	.byte	1                       #   On action: 1
	.byte	1                       # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                       #   No further actions
                                        # >> Catch TypeInfos <<
	.long	0                       # TypeInfo 1
	.p2align	2
                                        # -- End function
	.section	.text._ZNKSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8key_compEv,"axG",@progbits,_ZNKSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8key_compEv,comdat
	.weak	_ZNKSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8key_compEv # -- Begin function _ZNKSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8key_compEv
	.p2align	4, 0x90
	.type	_ZNKSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8key_compEv,@function
_ZNKSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8key_compEv: # @_ZNKSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8key_compEv
.Lfunc_begin102:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	popq	%rbp
	retq
.Lfunc_end102:
	.size	_ZNKSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8key_compEv, .Lfunc_end102-_ZNKSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8key_compEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11lower_boundERS7_,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11lower_boundERS7_,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11lower_boundERS7_ # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11lower_boundERS7_
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11lower_boundERS7_,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11lower_boundERS7_: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11lower_boundERS7_
.Lfunc_begin103:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	movq	%rsi, -16(%rbp)         # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_beginEv
	movq	-8(%rbp), %rdi          # 8-byte Reload
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_M_endEv
	movq	-8(%rbp), %rdi          # 8-byte Reload
	movq	-24(%rbp), %rsi         # 8-byte Reload
	movq	%rax, %rdx
	movq	-16(%rbp), %rcx         # 8-byte Reload
	addq	$32, %rsp
	popq	%rbp
	jmp	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_lower_boundEPSt13_Rb_tree_nodeISA_EPSt18_Rb_tree_node_baseRS7_ # TAILCALL
.Lfunc_end103:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11lower_boundERS7_, .Lfunc_end103-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE11lower_boundERS7_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_lower_boundEPSt13_Rb_tree_nodeISA_EPSt18_Rb_tree_node_baseRS7_,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_lower_boundEPSt13_Rb_tree_nodeISA_EPSt18_Rb_tree_node_baseRS7_,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_lower_boundEPSt13_Rb_tree_nodeISA_EPSt18_Rb_tree_node_baseRS7_ # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_lower_boundEPSt13_Rb_tree_nodeISA_EPSt18_Rb_tree_node_baseRS7_
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_lower_boundEPSt13_Rb_tree_nodeISA_EPSt18_Rb_tree_node_baseRS7_,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_lower_boundEPSt13_Rb_tree_nodeISA_EPSt18_Rb_tree_node_baseRS7_: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_lower_boundEPSt13_Rb_tree_nodeISA_EPSt18_Rb_tree_node_baseRS7_
.Lfunc_begin104:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$96, %rsp
	cmpq	$0, %rsi
	movq	%rdx, %rax
	movq	%rdx, -16(%rbp)         # 8-byte Spill
	movq	%rsi, -24(%rbp)         # 8-byte Spill
	movq	%rdi, -32(%rbp)         # 8-byte Spill
	movq	%rcx, -40(%rbp)         # 8-byte Spill
	movq	%rax, -48(%rbp)         # 8-byte Spill
	je	.LBB104_6
# %bb.1:
	movq	-24(%rbp), %rax         # 8-byte Reload
	movq	-16(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -56(%rbp)         # 8-byte Spill
	movq	%rcx, -64(%rbp)         # 8-byte Spill
.LBB104_2:                              # =>This Inner Loop Header: Depth=1
	movq	-64(%rbp), %rax         # 8-byte Reload
	movq	-56(%rbp), %rcx         # 8-byte Reload
	movq	%rcx, %rdi
	movq	%rax, -72(%rbp)         # 8-byte Spill
	movq	%rcx, -80(%rbp)         # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt13_Rb_tree_nodeISA_E
	movq	-32(%rbp), %rdi         # 8-byte Reload
	movq	%rax, %rsi
	movq	-40(%rbp), %rdx         # 8-byte Reload
	callq	_ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_
	testb	$1, %al
	jne	.LBB104_4
# %bb.3:                                #   in Loop: Header=BB104_2 Depth=1
	movq	-80(%rbp), %rdi         # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE7_S_leftEPSt18_Rb_tree_node_base
	movq	-80(%rbp), %rdi         # 8-byte Reload
	movq	%rdi, -88(%rbp)         # 8-byte Spill
	movq	%rax, -96(%rbp)         # 8-byte Spill
	jmp	.LBB104_5
.LBB104_4:                              #   in Loop: Header=BB104_2 Depth=1
	movq	-80(%rbp), %rdi         # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_S_rightEPSt18_Rb_tree_node_base
	movq	-72(%rbp), %rdi         # 8-byte Reload
	movq	%rdi, -88(%rbp)         # 8-byte Spill
	movq	%rax, -96(%rbp)         # 8-byte Spill
.LBB104_5:                              #   in Loop: Header=BB104_2 Depth=1
	movq	-96(%rbp), %rax         # 8-byte Reload
	movq	-88(%rbp), %rcx         # 8-byte Reload
	cmpq	$0, %rax
	movq	%rcx, %rdx
	movq	%rdx, -64(%rbp)         # 8-byte Spill
	movq	%rax, -56(%rbp)         # 8-byte Spill
	movq	%rcx, -48(%rbp)         # 8-byte Spill
	jne	.LBB104_2
.LBB104_6:
	movq	-48(%rbp), %rax         # 8-byte Reload
	leaq	-8(%rbp), %rdi
	movq	%rax, %rsi
	callq	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2EPSt18_Rb_tree_node_base
	movq	-8(%rbp), %rax
	addq	$96, %rsp
	popq	%rbp
	retq
.Lfunc_end104:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_lower_boundEPSt13_Rb_tree_nodeISA_EPSt18_Rb_tree_node_baseRS7_, .Lfunc_end104-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_lower_boundEPSt13_Rb_tree_nodeISA_EPSt18_Rb_tree_node_baseRS7_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE3endEv,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE3endEv,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE3endEv # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE3endEv
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE3endEv,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE3endEv: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE3endEv
.Lfunc_begin105:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	leaq	-8(%rbp), %rax
	addq	$8, %rdi
	movq	%rdi, -16(%rbp)         # 8-byte Spill
	movq	%rax, %rdi
	movq	-16(%rbp), %rsi         # 8-byte Reload
	callq	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEC2EPSt18_Rb_tree_node_base
	movq	-8(%rbp), %rax
	addq	$16, %rsp
	popq	%rbp
	retq
.Lfunc_end105:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE3endEv, .Lfunc_end105-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE3endEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE4findERS7_,"axG",@progbits,_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE4findERS7_,comdat
	.weak	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE4findERS7_ # -- Begin function _ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE4findERS7_
	.p2align	4, 0x90
	.type	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE4findERS7_,@function
_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE4findERS7_: # @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE4findERS7_
.Lfunc_begin106:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	leaq	-8(%rbp), %rax
	movq	%rdi, -24(%rbp)         # 8-byte Spill
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rsi, -40(%rbp)         # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_beginEv
	movq	-24(%rbp), %rdi         # 8-byte Reload
	movq	%rax, -48(%rbp)         # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_M_endEv
	movq	-24(%rbp), %rdi         # 8-byte Reload
	movq	-48(%rbp), %rsi         # 8-byte Reload
	movq	%rax, %rdx
	movq	-40(%rbp), %rcx         # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE14_M_lower_boundEPSt13_Rb_tree_nodeISA_EPSt18_Rb_tree_node_baseRS7_
	leaq	-16(%rbp), %rcx
	leaq	-8(%rbp), %rdx
	movq	%rax, -8(%rbp)
	movq	-24(%rbp), %rdi         # 8-byte Reload
	movq	%rdx, -56(%rbp)         # 8-byte Spill
	movq	%rcx, -64(%rbp)         # 8-byte Spill
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE3endEv
	leaq	-8(%rbp), %rdi
	leaq	-16(%rbp), %rsi
	movq	%rax, -16(%rbp)
	callq	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEeqERKSB_
	testb	$1, %al
	jne	.LBB106_2
# %bb.1:
	movq	-56(%rbp), %rax         # 8-byte Reload
	movq	(%rax), %rdi
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE6_S_keyEPKSt18_Rb_tree_node_base
	movq	-24(%rbp), %rdi         # 8-byte Reload
	movq	-40(%rbp), %rsi         # 8-byte Reload
	movq	%rax, %rdx
	callq	_ZNKSt4lessINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEclERKS5_S8_
	testb	$1, %al
	jne	.LBB106_2
	jmp	.LBB106_3
.LBB106_2:
	movq	-24(%rbp), %rdi         # 8-byte Reload
	callq	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE3endEv
	movq	%rax, -72(%rbp)         # 8-byte Spill
	jmp	.LBB106_4
.LBB106_3:
	movq	-56(%rbp), %rax         # 8-byte Reload
	movq	(%rax), %rcx
	movq	%rcx, -72(%rbp)         # 8-byte Spill
.LBB106_4:
	movq	-72(%rbp), %rax         # 8-byte Reload
	addq	$80, %rsp
	popq	%rbp
	retq
.Lfunc_end106:
	.size	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE4findERS7_, .Lfunc_end106-_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE4findERS7_
	.cfi_endproc
                                        # -- End function
	.text
	.globl	trace_logger_register_labelmap # -- Begin function trace_logger_register_labelmap
	.p2align	4, 0x90
	.type	trace_logger_register_labelmap,@function
trace_logger_register_labelmap:         # @trace_logger_register_labelmap
.Lfunc_begin107:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	$_Z12labelmap_strB5cxx11, %eax
	movl	%eax, %ecx
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	movq	%rcx, %rdi
	movq	-8(%rbp), %rcx          # 8-byte Reload
	movq	%rsi, -16(%rbp)         # 8-byte Spill
	movq	%rcx, %rsi
	movq	-16(%rbp), %rdx         # 8-byte Reload
	popq	%rbp
	jmp	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEPKcm # TAILCALL
.Lfunc_end107:
	.size	trace_logger_register_labelmap, .Lfunc_end107-trace_logger_register_labelmap
	.cfi_endproc
                                        # -- End function
	.globl	trace_logger_init       # -- Begin function trace_logger_init
	.p2align	4, 0x90
	.type	trace_logger_init,@function
trace_logger_init:                      # @trace_logger_init
.Lfunc_begin108:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movabsq	$lock, %rdi
	xorl	%eax, %eax
	movl	%eax, %esi
	callq	pthread_mutex_init
	cmpl	$0, %eax
	je	.LBB108_2
# %bb.1:
	movabsq	$.L.str.9.79, %rdi
	callq	perror
	movl	$4294967295, %edi       # imm = 0xFFFFFFFF
	callq	exit
.LBB108_2:
	movq	default_trace_name(%rip), %rdi
	callq	_Z12create_tracePKc
	movl	$_Z8fin_mainv, %eax
	movl	%eax, %edi
	popq	%rbp
	jmp	atexit                  # TAILCALL
.Lfunc_end108:
	.size	trace_logger_init, .Lfunc_end108-trace_logger_init
	.cfi_endproc
                                        # -- End function
	.globl	_Z8fin_mainv            # -- Begin function _Z8fin_mainv
	.p2align	4, 0x90
	.type	_Z8fin_mainv,@function
_Z8fin_mainv:                           # @_Z8fin_mainv
.Lfunc_begin109:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	callq	_ZTW5trace
	cmpq	$0, (%rax)
	je	.LBB109_2
# %bb.1:
	callq	_Z12fin_toplevelv
.LBB109_2:
	movabsq	$_Z8gz_filesB5cxx11, %rdi
	leaq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE5beginEv
	movabsq	$_Z8gz_filesB5cxx11, %rdi
	leaq	-16(%rbp), %rcx
	movq	%rax, -8(%rbp)
	movq	%rcx, -32(%rbp)         # 8-byte Spill
	callq	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE3endEv
	leaq	-8(%rbp), %rdi
	leaq	-16(%rbp), %rcx
	movq	%rax, -16(%rbp)
	movq	%rcx, %rsi
	movq	%rcx, -40(%rbp)         # 8-byte Spill
	callq	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEneERKSB_
	testb	$1, %al
	jne	.LBB109_3
	jmp	.LBB109_4
.LBB109_3:
	jmp	.LBB109_5
.LBB109_4:
	addq	$64, %rsp
	popq	%rbp
	retq
.LBB109_5:                              # =>This Inner Loop Header: Depth=1
	leaq	-8(%rbp), %rdi
	callq	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEptEv
	movq	32(%rax), %rdi
	callq	gzclose
	leaq	-8(%rbp), %rdi
	movl	%eax, -44(%rbp)         # 4-byte Spill
	callq	_ZNSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEppEv
	movabsq	$_Z8gz_filesB5cxx11, %rdi
	movq	%rax, -56(%rbp)         # 8-byte Spill
	callq	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE3endEv
	leaq	-8(%rbp), %rdi
	leaq	-16(%rbp), %rsi
	movq	-40(%rbp), %rcx         # 8-byte Reload
	movq	%rax, (%rcx)
	callq	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEneERKSB_
	testb	$1, %al
	jne	.LBB109_5
	jmp	.LBB109_4
.Lfunc_end109:
	.size	_Z8fin_mainv, .Lfunc_end109-_Z8fin_mainv
	.cfi_endproc
                                        # -- End function
	.globl	_Z12fin_toplevelv       # -- Begin function _Z12fin_toplevelv
	.p2align	4, 0x90
	.type	_Z12fin_toplevelv,@function
_Z12fin_toplevelv:                      # @_Z12fin_toplevelv
.Lfunc_begin110:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	callq	_ZTW5trace
	movq	(%rax), %rax
	cmpq	$0, %rax
	movq	%rax, -8(%rbp)          # 8-byte Spill
	je	.LBB110_2
# %bb.1:
	movq	-8(%rbp), %rdi          # 8-byte Reload
	callq	_ZN10trace_infoD2Ev
	movq	-8(%rbp), %rdi          # 8-byte Reload
	callq	_ZdlPv
.LBB110_2:
	callq	_ZTW5trace
	movq	$0, (%rax)
	addq	$16, %rsp
	popq	%rbp
	retq
.Lfunc_end110:
	.size	_Z12fin_toplevelv, .Lfunc_end110-_Z12fin_toplevelv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE5beginEv,"axG",@progbits,_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE5beginEv,comdat
	.weak	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE5beginEv # -- Begin function _ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE5beginEv
	.p2align	4, 0x90
	.type	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE5beginEv,@function
_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE5beginEv: # @_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE5beginEv
.Lfunc_begin111:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	jmp	_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_P8gzFile_sESt10_Select1stISA_ESt4lessIS5_ESaISA_EE5beginEv # TAILCALL
.Lfunc_end111:
	.size	_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE5beginEv, .Lfunc_end111-_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sSt4lessIS5_ESaISt4pairIKS5_S7_EEE5beginEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEptEv,"axG",@progbits,_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEptEv,comdat
	.weak	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEptEv # -- Begin function _ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEptEv
	.p2align	4, 0x90
	.type	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEptEv,@function
_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEptEv: # @_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEptEv
.Lfunc_begin112:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	(%rdi), %rdi
	popq	%rbp
	jmp	_ZNSt13_Rb_tree_nodeISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEE9_M_valptrEv # TAILCALL
.Lfunc_end112:
	.size	_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEptEv, .Lfunc_end112-_ZNKSt17_Rb_tree_iteratorISt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP8gzFile_sEEptEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN10trace_infoD2Ev,"axG",@progbits,_ZN10trace_infoD2Ev,comdat
	.weak	_ZN10trace_infoD2Ev     # -- Begin function _ZN10trace_infoD2Ev
	.p2align	4, 0x90
	.type	_ZN10trace_infoD2Ev,@function
_ZN10trace_infoD2Ev:                    # @_ZN10trace_infoD2Ev
.Lfunc_begin113:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, %rax
	addq	$48, %rax
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	movq	%rax, %rdi
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev
	movq	-8(%rbp), %rax          # 8-byte Reload
	movq	%rax, %rdi
	addq	$16, %rsp
	popq	%rbp
	jmp	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev # TAILCALL
.Lfunc_end113:
	.size	_ZN10trace_infoD2Ev, .Lfunc_end113-_ZN10trace_infoD2Ev
	.cfi_endproc
                                        # -- End function
	.text
	.globl	llvmtracer_set_trace_name # -- Begin function llvmtracer_set_trace_name
	.p2align	4, 0x90
	.type	llvmtracer_set_trace_name,@function
llvmtracer_set_trace_name:              # @llvmtracer_set_trace_name
.Lfunc_begin114:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	callq	_ZTW5trace
	cmpq	$0, (%rax)
	jne	.LBB114_2
# %bb.1:
	movq	-8(%rbp), %rdi          # 8-byte Reload
	callq	_Z12create_tracePKc
	jmp	.LBB114_3
.LBB114_2:
	callq	_ZTW5trace
	movq	(%rax), %rdi
	movq	-8(%rbp), %rsi          # 8-byte Reload
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEPKc
	movq	%rax, -16(%rbp)         # 8-byte Spill
.LBB114_3:
	addq	$16, %rsp
	popq	%rbp
	retq
.Lfunc_end114:
	.size	llvmtracer_set_trace_name, .Lfunc_end114-llvmtracer_set_trace_name
	.cfi_endproc
                                        # -- End function
	.globl	_Z10log_or_notbbiPc     # -- Begin function _Z10log_or_notbbiPc
	.p2align	4, 0x90
	.type	_Z10log_or_notbbiPc,@function
_Z10log_or_notbbiPc:                    # @_Z10log_or_notbbiPc
.Lfunc_begin115:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movb	%sil, %al
	movb	%dil, %r8b
	testb	$1, %r8b
	movl	%edx, -4(%rbp)          # 4-byte Spill
	movq	%rcx, -16(%rbp)         # 8-byte Spill
	movb	%al, -17(%rbp)          # 1-byte Spill
	jne	.LBB115_2
	jmp	.LBB115_1
.LBB115_1:
	movb	-17(%rbp), %al          # 1-byte Reload
	xorb	$-1, %al
	andb	$1, %al
	movzbl	%al, %ecx
	movl	%ecx, -24(%rbp)         # 4-byte Spill
	jmp	.LBB115_9
.LBB115_2:
	movb	-17(%rbp), %al          # 1-byte Reload
	testb	$1, %al
	jne	.LBB115_4
	jmp	.LBB115_3
.LBB115_3:
	callq	_ZTW5trace
	movq	(%rax), %rax
	movl	80(%rax), %ecx
	movl	%ecx, -24(%rbp)         # 4-byte Spill
	jmp	.LBB115_9
.LBB115_4:
	xorl	%eax, %eax
	movl	-4(%rbp), %ecx          # 4-byte Reload
	cmpl	$1, %ecx
	movl	%eax, -24(%rbp)         # 4-byte Spill
	jne	.LBB115_9
# %bb.5:
	callq	_ZTW5trace
	movq	(%rax), %rax
	addq	$48, %rax
	movq	%rax, %rdi
	callq	_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv
	cmpq	$0, %rax
	jne	.LBB115_7
# %bb.6:
	movabsq	$.L.str.10.80, %rdi
	movabsq	$.L.str.4.73, %rsi
	movl	$132, %edx
	movabsq	$.L__PRETTY_FUNCTION__._Z10log_or_notbbiPc, %rcx
	callq	__assert_fail
.LBB115_7:
	callq	_ZTW5trace
	movq	(%rax), %rax
	addq	$48, %rax
	movq	%rax, %rdi
	callq	_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5c_strEv
	movl	$1, %ecx
	movq	-16(%rbp), %rdi         # 8-byte Reload
	movq	%rax, %rsi
	movl	%ecx, -28(%rbp)         # 4-byte Spill
	callq	strcmp
	cmpl	$0, %eax
	movl	-28(%rbp), %eax         # 4-byte Reload
	movl	%eax, -24(%rbp)         # 4-byte Spill
	je	.LBB115_9
# %bb.8:
	movabsq	$.L.str.11.81, %rdi
	movabsq	$.L.str.4.73, %rsi
	movl	$137, %edx
	movabsq	$.L__PRETTY_FUNCTION__._Z10log_or_notbbiPc, %rcx
	callq	__assert_fail
.LBB115_9:
	movl	-24(%rbp), %eax         # 4-byte Reload
	addq	$32, %rsp
	popq	%rbp
	retq
.Lfunc_end115:
	.size	_Z10log_or_notbbiPc, .Lfunc_end115-_Z10log_or_notbbiPc
	.cfi_endproc
                                        # -- End function
	.globl	_Z20convert_bytes_to_hexPcPhi # -- Begin function _Z20convert_bytes_to_hexPcPhi
	.p2align	4, 0x90
	.type	_Z20convert_bytes_to_hexPcPhi,@function
_Z20convert_bytes_to_hexPcPhi:          # @_Z20convert_bytes_to_hexPcPhi
.Lfunc_begin116:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movw	.L.str.12.82, %ax
	movw	%ax, (%rdi)
	movb	.L.str.12.82+2, %cl
	movb	%cl, 2(%rdi)
	addq	$2, %rdi
	cmpl	$0, %edx
	movq	%rdi, %r8
	movq	%rsi, -8(%rbp)          # 8-byte Spill
	movl	%edx, -12(%rbp)         # 4-byte Spill
	movq	%rdi, -24(%rbp)         # 8-byte Spill
	movq	%r8, -32(%rbp)          # 8-byte Spill
	jle	.LBB116_2
# %bb.1:
	xorl	%eax, %eax
	movl	%eax, %ecx
	movl	-12(%rbp), %eax         # 4-byte Reload
	movl	%eax, %edx
	movl	%edx, %esi
	movq	-24(%rbp), %rdi         # 8-byte Reload
	movq	%rsi, -40(%rbp)         # 8-byte Spill
	movq	%rcx, -48(%rbp)         # 8-byte Spill
	movq	%rdi, -56(%rbp)         # 8-byte Spill
	jmp	.LBB116_3
.LBB116_2:
	movq	-32(%rbp), %rax         # 8-byte Reload
	movb	$0, (%rax)
	addq	$80, %rsp
	popq	%rbp
	retq
.LBB116_3:                              # =>This Inner Loop Header: Depth=1
	movq	-56(%rbp), %rax         # 8-byte Reload
	movq	-48(%rbp), %rcx         # 8-byte Reload
	movabsq	$.L.str.13.83, %rsi
	movq	-8(%rbp), %rdx          # 8-byte Reload
	movzbl	(%rdx,%rcx), %edx
	movq	%rax, %rdi
	movq	%rax, -64(%rbp)         # 8-byte Spill
	movb	$0, %al
	movq	%rcx, -72(%rbp)         # 8-byte Spill
	callq	sprintf
	movslq	%eax, %rcx
	movq	-64(%rbp), %rsi         # 8-byte Reload
	addq	%rcx, %rsi
	movq	-72(%rbp), %rcx         # 8-byte Reload
	addq	$1, %rcx
	movq	-40(%rbp), %rdi         # 8-byte Reload
	cmpq	%rdi, %rcx
	movq	%rsi, %r8
	movq	%rcx, -48(%rbp)         # 8-byte Spill
	movq	%r8, -32(%rbp)          # 8-byte Spill
	movq	%rsi, -56(%rbp)         # 8-byte Spill
	je	.LBB116_2
	jmp	.LBB116_3
.Lfunc_end116:
	.size	_Z20convert_bytes_to_hexPcPhi, .Lfunc_end116-_Z20convert_bytes_to_hexPcPhi
	.cfi_endproc
                                        # -- End function
	.globl	trace_logger_update_status # -- Begin function trace_logger_update_status
	.p2align	4, 0x90
	.type	trace_logger_update_status,@function
trace_logger_update_status:             # @trace_logger_update_status
.Lfunc_begin117:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movb	%cl, %al
	movb	%dl, %r8b
	movl	%esi, -4(%rbp)          # 4-byte Spill
	movb	%al, -5(%rbp)           # 1-byte Spill
	movb	%r8b, -6(%rbp)          # 1-byte Spill
	movq	%rdi, -16(%rbp)         # 8-byte Spill
	callq	_ZTW5trace
	cmpq	$0, (%rax)
	jne	.LBB117_3
# %bb.1:
	movb	-6(%rbp), %al           # 1-byte Reload
	testb	$1, %al
	jne	.LBB117_2
	jmp	.LBB117_13
.LBB117_2:
	movq	default_trace_name, %rdi
	callq	_Z12create_tracePKc
.LBB117_3:
	callq	_ZTW5trace
	movq	(%rax), %rax
	movl	80(%rax), %ecx
	movb	-5(%rbp), %dl           # 1-byte Reload
	andb	$1, %dl
	movzbl	%dl, %edi
	movb	-6(%rbp), %dl           # 1-byte Reload
	andb	$1, %dl
	movzbl	%dl, %esi
	movl	-4(%rbp), %edx          # 4-byte Reload
	movq	-16(%rbp), %rax         # 8-byte Reload
	movl	%ecx, -20(%rbp)         # 4-byte Spill
	movq	%rax, %rcx
	callq	_Z10log_or_notbbiPc
	movl	%eax, -24(%rbp)         # 4-byte Spill
	callq	_ZTW5trace
	movq	(%rax), %rax
	movl	-24(%rbp), %edx         # 4-byte Reload
	movl	%edx, 80(%rax)
	movl	-20(%rbp), %esi         # 4-byte Reload
	testl	%esi, %esi
	je	.LBB117_4
	jmp	.LBB117_14
.LBB117_14:
	movl	-20(%rbp), %eax         # 4-byte Reload
	subl	$1, %eax
	movl	%eax, -28(%rbp)         # 4-byte Spill
	je	.LBB117_6
	jmp	.LBB117_8
.LBB117_4:
	callq	_ZTW5trace
	movq	(%rax), %rax
	cmpl	$1, 80(%rax)
	jne	.LBB117_8
# %bb.5:
	callq	_ZTW5trace
	movq	(%rax), %rdi
	callq	_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5c_strEv
	movq	%rax, -40(%rbp)         # 8-byte Spill
	callq	_ZTW5trace
	movabsq	$.L.str.14.86, %rdi
	movq	(%rax), %rax
	movq	40(%rax), %rdx
	movq	-40(%rbp), %rsi         # 8-byte Reload
	movb	$0, %al
	callq	printf
	movq	stdout, %rdi
	movl	%eax, -44(%rbp)         # 4-byte Spill
	callq	fflush
	movl	%eax, -48(%rbp)         # 4-byte Spill
	jmp	.LBB117_8
.LBB117_6:
	callq	_ZTW5trace
	movq	(%rax), %rax
	cmpl	$1, 80(%rax)
	je	.LBB117_8
# %bb.7:
	callq	_ZTW5trace
	movq	(%rax), %rdi
	callq	_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5c_strEv
	movq	%rax, -56(%rbp)         # 8-byte Spill
	callq	_ZTW5trace
	movabsq	$.L.str.15.87, %rdi
	movq	(%rax), %rax
	movq	40(%rax), %rdx
	movq	-56(%rbp), %rsi         # 8-byte Reload
	movb	$0, %al
	callq	printf
	movq	stdout, %rdi
	movl	%eax, -60(%rbp)         # 4-byte Spill
	callq	fflush
	movl	%eax, -64(%rbp)         # 4-byte Spill
.LBB117_8:
	callq	_ZTW5trace
	movq	(%rax), %rax
	addq	$48, %rax
	movq	%rax, %rdi
	callq	_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv
	cmpq	$0, %rax
	jne	.LBB117_11
# %bb.9:
	callq	_ZTW5trace
	movq	(%rax), %rax
	cmpl	$0, 80(%rax)
	jne	.LBB117_11
# %bb.10:
	callq	_ZTW5trace
	movq	(%rax), %rax
	addq	$48, %rax
	movq	%rax, %rdi
	movq	-16(%rbp), %rsi         # 8-byte Reload
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEPKc
	movq	%rax, -72(%rbp)         # 8-byte Spill
	jmp	.LBB117_13
.LBB117_11:
	callq	_ZTW5trace
	movq	(%rax), %rax
	cmpl	$1, 80(%rax)
	jne	.LBB117_13
# %bb.12:
	callq	_ZTW5trace
	movabsq	$.L.str.16.88, %rsi
	movq	(%rax), %rax
	addq	$48, %rax
	movq	%rax, %rdi
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEPKc
	movq	%rax, -80(%rbp)         # 8-byte Spill
	callq	_Z12fin_toplevelv
.LBB117_13:
	addq	$80, %rsp
	popq	%rbp
	retq
.Lfunc_end117:
	.size	trace_logger_update_status, .Lfunc_end117-trace_logger_update_status
	.cfi_endproc
                                        # -- End function
	.globl	_Z10do_not_logv         # -- Begin function _Z10do_not_logv
	.p2align	4, 0x90
	.type	_Z10do_not_logv,@function
_Z10do_not_logv:                        # @_Z10do_not_logv
.Lfunc_begin118:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	callq	_ZTW5trace
	movb	$1, %cl
	cmpq	$0, (%rax)
	movb	%cl, -1(%rbp)           # 1-byte Spill
	je	.LBB118_2
# %bb.1:
	callq	_ZTW5trace
	movq	(%rax), %rax
	cmpl	$1, 80(%rax)
	sete	%cl
	movb	%cl, -1(%rbp)           # 1-byte Spill
.LBB118_2:
	movb	-1(%rbp), %al           # 1-byte Reload
	andb	$1, %al
	movzbl	%al, %eax
	addq	$16, %rsp
	popq	%rbp
	retq
.Lfunc_end118:
	.size	_Z10do_not_logv, .Lfunc_end118-_Z10do_not_logv
	.cfi_endproc
                                        # -- End function
	.globl	trace_logger_log_entry  # -- Begin function trace_logger_log_entry
	.p2align	4, 0x90
	.type	trace_logger_log_entry,@function
trace_logger_log_entry:                 # @trace_logger_log_entry
.Lfunc_begin119:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)          # 8-byte Spill
	movl	%esi, -12(%rbp)         # 4-byte Spill
	callq	_ZTW5trace
	cmpq	$0, (%rax)
	jne	.LBB119_2
# %bb.1:
	movq	default_trace_name, %rdi
	callq	_Z12create_tracePKc
.LBB119_2:
	callq	_Z10do_not_logv
	testb	$1, %al
	jne	.LBB119_4
# %bb.3:
	callq	_Z15open_trace_filev
	callq	_ZTW5trace
	movabsq	$.L.str.17.91, %rsi
	movq	(%rax), %rax
	movq	32(%rax), %rdi
	movq	-8(%rbp), %rdx          # 8-byte Reload
	movl	-12(%rbp), %ecx         # 4-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -16(%rbp)         # 4-byte Spill
.LBB119_4:
	addq	$16, %rsp
	popq	%rbp
	retq
.Lfunc_end119:
	.size	trace_logger_log_entry, .Lfunc_end119-trace_logger_log_entry
	.cfi_endproc
                                        # -- End function
	.globl	trace_logger_log0       # -- Begin function trace_logger_log0
	.p2align	4, 0x90
	.type	trace_logger_log0,@function
trace_logger_log0:                      # @trace_logger_log0
.Lfunc_begin120:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movq	%rsi, -8(%rbp)          # 8-byte Spill
	movl	%r8d, -12(%rbp)         # 4-byte Spill
	movq	%rcx, -24(%rbp)         # 8-byte Spill
	movq	%rdx, -32(%rbp)         # 8-byte Spill
	movl	%edi, -36(%rbp)         # 4-byte Spill
	callq	_ZTW5trace
	cmpq	$0, (%rax)
	je	.LBB120_3
# %bb.1:
	callq	_Z10do_not_logv
	testb	$1, %al
	jne	.LBB120_3
# %bb.2:
	callq	_ZTW5trace
	movq	(%rax), %rax
	movq	32(%rax), %rdi
	movq	%rdi, -48(%rbp)         # 8-byte Spill
	callq	_ZTW5trace
	movabsq	$.L.str.18.94, %rsi
	movq	(%rax), %rax
	movq	40(%rax), %rax
	movq	-48(%rbp), %rdi         # 8-byte Reload
	movl	-36(%rbp), %edx         # 4-byte Reload
	movq	-8(%rbp), %rcx          # 8-byte Reload
	movq	-32(%rbp), %r8          # 8-byte Reload
	movq	-24(%rbp), %r9          # 8-byte Reload
	movl	-12(%rbp), %r10d        # 4-byte Reload
	movl	%r10d, (%rsp)
	movq	%rax, 8(%rsp)
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -52(%rbp)         # 4-byte Spill
	callq	_ZTW5trace
	movq	(%rax), %rax
	movq	40(%rax), %rcx
	addq	$1, %rcx
	movq	%rcx, 40(%rax)
.LBB120_3:
	addq	$80, %rsp
	popq	%rbp
	retq
.Lfunc_end120:
	.size	trace_logger_log0, .Lfunc_end120-trace_logger_log0
	.cfi_endproc
                                        # -- End function
	.globl	trace_logger_log_int    # -- Begin function trace_logger_log_int
	.p2align	4, 0x90
	.type	trace_logger_log_int,@function
trace_logger_log_int:                   # @trace_logger_log_int
.Lfunc_begin121:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$96, %rsp
	movq	16(%rbp), %rax
	movq	%rax, -8(%rbp)          # 8-byte Spill
	movl	%r9d, -12(%rbp)         # 4-byte Spill
	movq	%r8, -24(%rbp)          # 8-byte Spill
	movl	%ecx, -28(%rbp)         # 4-byte Spill
	movq	%rdx, -40(%rbp)         # 8-byte Spill
	movl	%esi, -44(%rbp)         # 4-byte Spill
	movl	%edi, -48(%rbp)         # 4-byte Spill
	callq	_ZTW5trace
	cmpq	$0, (%rax)
	je	.LBB121_12
# %bb.1:
	callq	_Z10do_not_logv
	testb	$1, %al
	jne	.LBB121_12
# %bb.2:
	callq	_ZTW5trace
	movq	(%rax), %rax
	movq	32(%rax), %rax
	movl	-48(%rbp), %ecx         # 4-byte Reload
	subl	$19134, %ecx            # imm = 0x4ABE
	movq	%rax, -56(%rbp)         # 8-byte Spill
	movl	%ecx, -60(%rbp)         # 4-byte Spill
	je	.LBB121_3
	jmp	.LBB121_13
.LBB121_13:
	movl	-48(%rbp), %eax         # 4-byte Reload
	subl	$24601, %eax            # imm = 0x6019
	movl	%eax, -64(%rbp)         # 4-byte Spill
	je	.LBB121_4
	jmp	.LBB121_5
.LBB121_3:
	movabsq	$.L.str.19.97, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movl	-44(%rbp), %edx         # 4-byte Reload
	movq	-40(%rbp), %rcx         # 8-byte Reload
	movl	-28(%rbp), %r8d         # 4-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -68(%rbp)         # 4-byte Spill
	jmp	.LBB121_6
.LBB121_4:
	movabsq	$.L.str.20.98, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movl	-44(%rbp), %edx         # 4-byte Reload
	movq	-40(%rbp), %rcx         # 8-byte Reload
	movl	-28(%rbp), %r8d         # 4-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -72(%rbp)         # 4-byte Spill
	jmp	.LBB121_6
.LBB121_5:
	movabsq	$.L.str.21.99, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movl	-48(%rbp), %edx         # 4-byte Reload
	movl	-44(%rbp), %ecx         # 4-byte Reload
	movq	-40(%rbp), %r8          # 8-byte Reload
	movl	-28(%rbp), %r9d         # 4-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -76(%rbp)         # 4-byte Spill
.LBB121_6:
	movl	-28(%rbp), %eax         # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB121_8
# %bb.7:
	movabsq	$.L.str.22.100, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movq	-24(%rbp), %rdx         # 8-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -80(%rbp)         # 4-byte Spill
	jmp	.LBB121_9
.LBB121_8:
	movabsq	$.L.str.23.101, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -84(%rbp)         # 4-byte Spill
.LBB121_9:
	movl	-12(%rbp), %eax         # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB121_11
# %bb.10:
	movabsq	$.L.str.24.102, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movq	-8(%rbp), %rdx          # 8-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -88(%rbp)         # 4-byte Spill
	jmp	.LBB121_12
.LBB121_11:
	movabsq	$.L.str.25.103, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -92(%rbp)         # 4-byte Spill
.LBB121_12:
	addq	$96, %rsp
	popq	%rbp
	retq
.Lfunc_end121:
	.size	trace_logger_log_int, .Lfunc_end121-trace_logger_log_int
	.cfi_endproc
                                        # -- End function
	.globl	trace_logger_log_ptr    # -- Begin function trace_logger_log_ptr
	.p2align	4, 0x90
	.type	trace_logger_log_ptr,@function
trace_logger_log_ptr:                   # @trace_logger_log_ptr
.Lfunc_begin122:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$96, %rsp
	movq	16(%rbp), %rax
	movq	%rax, -8(%rbp)          # 8-byte Spill
	movl	%r9d, -12(%rbp)         # 4-byte Spill
	movq	%r8, -24(%rbp)          # 8-byte Spill
	movl	%ecx, -28(%rbp)         # 4-byte Spill
	movq	%rdx, -40(%rbp)         # 8-byte Spill
	movl	%esi, -44(%rbp)         # 4-byte Spill
	movl	%edi, -48(%rbp)         # 4-byte Spill
	callq	_ZTW5trace
	cmpq	$0, (%rax)
	je	.LBB122_12
# %bb.1:
	callq	_Z10do_not_logv
	testb	$1, %al
	jne	.LBB122_12
# %bb.2:
	callq	_ZTW5trace
	movq	(%rax), %rax
	movq	32(%rax), %rax
	movl	-48(%rbp), %ecx         # 4-byte Reload
	subl	$19134, %ecx            # imm = 0x4ABE
	movq	%rax, -56(%rbp)         # 8-byte Spill
	movl	%ecx, -60(%rbp)         # 4-byte Spill
	je	.LBB122_3
	jmp	.LBB122_13
.LBB122_13:
	movl	-48(%rbp), %eax         # 4-byte Reload
	subl	$24601, %eax            # imm = 0x6019
	movl	%eax, -64(%rbp)         # 4-byte Spill
	je	.LBB122_4
	jmp	.LBB122_5
.LBB122_3:
	movabsq	$.L.str.26.106, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movl	-44(%rbp), %edx         # 4-byte Reload
	movq	-40(%rbp), %rcx         # 8-byte Reload
	movl	-28(%rbp), %r8d         # 4-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -68(%rbp)         # 4-byte Spill
	jmp	.LBB122_6
.LBB122_4:
	movabsq	$.L.str.27.107, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movl	-44(%rbp), %edx         # 4-byte Reload
	movq	-40(%rbp), %rcx         # 8-byte Reload
	movl	-28(%rbp), %r8d         # 4-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -72(%rbp)         # 4-byte Spill
	jmp	.LBB122_6
.LBB122_5:
	movabsq	$.L.str.28.108, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movl	-48(%rbp), %edx         # 4-byte Reload
	movl	-44(%rbp), %ecx         # 4-byte Reload
	movq	-40(%rbp), %r8          # 8-byte Reload
	movl	-28(%rbp), %r9d         # 4-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -76(%rbp)         # 4-byte Spill
.LBB122_6:
	movl	-28(%rbp), %eax         # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB122_8
# %bb.7:
	movabsq	$.L.str.22.100, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movq	-24(%rbp), %rdx         # 8-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -80(%rbp)         # 4-byte Spill
	jmp	.LBB122_9
.LBB122_8:
	movabsq	$.L.str.23.101, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -84(%rbp)         # 4-byte Spill
.LBB122_9:
	movl	-12(%rbp), %eax         # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB122_11
# %bb.10:
	movabsq	$.L.str.24.102, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movq	-8(%rbp), %rdx          # 8-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -88(%rbp)         # 4-byte Spill
	jmp	.LBB122_12
.LBB122_11:
	movabsq	$.L.str.25.103, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -92(%rbp)         # 4-byte Spill
.LBB122_12:
	addq	$96, %rsp
	popq	%rbp
	retq
.Lfunc_end122:
	.size	trace_logger_log_ptr, .Lfunc_end122-trace_logger_log_ptr
	.cfi_endproc
                                        # -- End function
	.globl	trace_logger_log_string # -- Begin function trace_logger_log_string
	.p2align	4, 0x90
	.type	trace_logger_log_string,@function
trace_logger_log_string:                # @trace_logger_log_string
.Lfunc_begin123:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$96, %rsp
	movq	16(%rbp), %rax
	movq	%rax, -8(%rbp)          # 8-byte Spill
	movl	%r9d, -12(%rbp)         # 4-byte Spill
	movq	%r8, -24(%rbp)          # 8-byte Spill
	movl	%ecx, -28(%rbp)         # 4-byte Spill
	movq	%rdx, -40(%rbp)         # 8-byte Spill
	movl	%esi, -44(%rbp)         # 4-byte Spill
	movl	%edi, -48(%rbp)         # 4-byte Spill
	callq	_ZTW5trace
	cmpq	$0, (%rax)
	je	.LBB123_12
# %bb.1:
	callq	_Z10do_not_logv
	testb	$1, %al
	jne	.LBB123_12
# %bb.2:
	callq	_ZTW5trace
	movq	(%rax), %rax
	movq	32(%rax), %rax
	movl	-48(%rbp), %ecx         # 4-byte Reload
	subl	$19134, %ecx            # imm = 0x4ABE
	movq	%rax, -56(%rbp)         # 8-byte Spill
	movl	%ecx, -60(%rbp)         # 4-byte Spill
	je	.LBB123_3
	jmp	.LBB123_13
.LBB123_13:
	movl	-48(%rbp), %eax         # 4-byte Reload
	subl	$24601, %eax            # imm = 0x6019
	movl	%eax, -64(%rbp)         # 4-byte Spill
	je	.LBB123_4
	jmp	.LBB123_5
.LBB123_3:
	movabsq	$.L.str.29.111, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movl	-44(%rbp), %edx         # 4-byte Reload
	movq	-40(%rbp), %rcx         # 8-byte Reload
	movl	-28(%rbp), %r8d         # 4-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -68(%rbp)         # 4-byte Spill
	jmp	.LBB123_6
.LBB123_4:
	movabsq	$.L.str.30.112, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movl	-44(%rbp), %edx         # 4-byte Reload
	movq	-40(%rbp), %rcx         # 8-byte Reload
	movl	-28(%rbp), %r8d         # 4-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -72(%rbp)         # 4-byte Spill
	jmp	.LBB123_6
.LBB123_5:
	movabsq	$.L.str.31.113, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movl	-48(%rbp), %edx         # 4-byte Reload
	movl	-44(%rbp), %ecx         # 4-byte Reload
	movq	-40(%rbp), %r8          # 8-byte Reload
	movl	-28(%rbp), %r9d         # 4-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -76(%rbp)         # 4-byte Spill
.LBB123_6:
	movl	-28(%rbp), %eax         # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB123_8
# %bb.7:
	movabsq	$.L.str.22.100, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movq	-24(%rbp), %rdx         # 8-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -80(%rbp)         # 4-byte Spill
	jmp	.LBB123_9
.LBB123_8:
	movabsq	$.L.str.23.101, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -84(%rbp)         # 4-byte Spill
.LBB123_9:
	movl	-12(%rbp), %eax         # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB123_11
# %bb.10:
	movabsq	$.L.str.24.102, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movq	-8(%rbp), %rdx          # 8-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -88(%rbp)         # 4-byte Spill
	jmp	.LBB123_12
.LBB123_11:
	movabsq	$.L.str.25.103, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -92(%rbp)         # 4-byte Spill
.LBB123_12:
	addq	$96, %rsp
	popq	%rbp
	retq
.Lfunc_end123:
	.size	trace_logger_log_string, .Lfunc_end123-trace_logger_log_string
	.cfi_endproc
                                        # -- End function
	.globl	trace_logger_log_double # -- Begin function trace_logger_log_double
	.p2align	4, 0x90
	.type	trace_logger_log_double,@function
trace_logger_log_double:                # @trace_logger_log_double
.Lfunc_begin124:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$96, %rsp
	movl	%r8d, -4(%rbp)          # 4-byte Spill
	movq	%rcx, -16(%rbp)         # 8-byte Spill
	movl	%edx, -20(%rbp)         # 4-byte Spill
	movsd	%xmm0, -32(%rbp)        # 8-byte Spill
	movl	%esi, -36(%rbp)         # 4-byte Spill
	movl	%edi, -40(%rbp)         # 4-byte Spill
	movq	%r9, -48(%rbp)          # 8-byte Spill
	callq	_ZTW5trace
	cmpq	$0, (%rax)
	je	.LBB124_12
# %bb.1:
	callq	_Z10do_not_logv
	testb	$1, %al
	jne	.LBB124_12
# %bb.2:
	callq	_ZTW5trace
	movq	(%rax), %rax
	movq	32(%rax), %rax
	movl	-40(%rbp), %ecx         # 4-byte Reload
	subl	$19134, %ecx            # imm = 0x4ABE
	movq	%rax, -56(%rbp)         # 8-byte Spill
	movl	%ecx, -60(%rbp)         # 4-byte Spill
	je	.LBB124_3
	jmp	.LBB124_13
.LBB124_13:
	movl	-40(%rbp), %eax         # 4-byte Reload
	subl	$24601, %eax            # imm = 0x6019
	movl	%eax, -64(%rbp)         # 4-byte Spill
	je	.LBB124_4
	jmp	.LBB124_5
.LBB124_3:
	movabsq	$.L.str.32.114, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movl	-36(%rbp), %edx         # 4-byte Reload
	movsd	-32(%rbp), %xmm0        # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movl	-20(%rbp), %ecx         # 4-byte Reload
	movb	$1, %al
	callq	gzprintf
	movl	%eax, -68(%rbp)         # 4-byte Spill
	jmp	.LBB124_6
.LBB124_4:
	movabsq	$.L.str.33.115, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movl	-36(%rbp), %edx         # 4-byte Reload
	movsd	-32(%rbp), %xmm0        # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movl	-20(%rbp), %ecx         # 4-byte Reload
	movb	$1, %al
	callq	gzprintf
	movl	%eax, -72(%rbp)         # 4-byte Spill
	jmp	.LBB124_6
.LBB124_5:
	movabsq	$.L.str.34.116, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movl	-40(%rbp), %edx         # 4-byte Reload
	movl	-36(%rbp), %ecx         # 4-byte Reload
	movsd	-32(%rbp), %xmm0        # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movl	-20(%rbp), %r8d         # 4-byte Reload
	movb	$1, %al
	callq	gzprintf
	movl	%eax, -76(%rbp)         # 4-byte Spill
.LBB124_6:
	movl	-20(%rbp), %eax         # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB124_8
# %bb.7:
	movabsq	$.L.str.22.100, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movq	-16(%rbp), %rdx         # 8-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -80(%rbp)         # 4-byte Spill
	jmp	.LBB124_9
.LBB124_8:
	movabsq	$.L.str.23.101, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -84(%rbp)         # 4-byte Spill
.LBB124_9:
	movl	-4(%rbp), %eax          # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB124_11
# %bb.10:
	movabsq	$.L.str.24.102, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movq	-48(%rbp), %rdx         # 8-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -88(%rbp)         # 4-byte Spill
	jmp	.LBB124_12
.LBB124_11:
	movabsq	$.L.str.25.103, %rsi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -92(%rbp)         # 4-byte Spill
.LBB124_12:
	addq	$96, %rsp
	popq	%rbp
	retq
.Lfunc_end124:
	.size	trace_logger_log_double, .Lfunc_end124-trace_logger_log_double
	.cfi_endproc
                                        # -- End function
	.globl	trace_logger_log_vector # -- Begin function trace_logger_log_vector
	.p2align	4, 0x90
	.type	trace_logger_log_vector,@function
trace_logger_log_vector:                # @trace_logger_log_vector
.Lfunc_begin125:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movq	16(%rbp), %rax
	movq	%rax, -8(%rbp)          # 8-byte Spill
	movl	%r9d, -12(%rbp)         # 4-byte Spill
	movq	%r8, -24(%rbp)          # 8-byte Spill
	movl	%ecx, -28(%rbp)         # 4-byte Spill
	movq	%rdx, -40(%rbp)         # 8-byte Spill
	movl	%esi, -44(%rbp)         # 4-byte Spill
	movl	%edi, -48(%rbp)         # 4-byte Spill
	callq	_ZTW5trace
	cmpq	$0, (%rax)
	je	.LBB125_13
# %bb.1:
	callq	_Z10do_not_logv
	testb	$1, %al
	jne	.LBB125_13
# %bb.2:
	movl	-44(%rbp), %eax         # 4-byte Reload
	sarl	$31, %eax
	movl	%eax, %ecx
	shrl	$30, %ecx
	movl	-44(%rbp), %edx         # 4-byte Reload
	addl	%ecx, %edx
	sarl	$2, %edx
	addl	$3, %edx
	movl	%edx, %esi
	movq	%rsp, %rdi
	addq	$15, %rsi
	movabsq	$8589934576, %r8        # imm = 0x1FFFFFFF0
	andq	%r8, %rsi
	movq	%rsp, %r8
	subq	%rsi, %r8
	movq	%r8, %rsp
	shrl	$29, %eax
	movl	-44(%rbp), %ecx         # 4-byte Reload
	addl	%eax, %ecx
	sarl	$3, %ecx
	movq	%rdi, -56(%rbp)         # 8-byte Spill
	movq	%r8, %rdi
	movq	-40(%rbp), %rsi         # 8-byte Reload
	movl	%ecx, %edx
	movq	%r8, -64(%rbp)          # 8-byte Spill
	callq	_Z20convert_bytes_to_hexPcPhi
	callq	_ZTW5trace
	movq	(%rax), %rax
	movq	32(%rax), %rax
	movl	-48(%rbp), %ecx         # 4-byte Reload
	subl	$19134, %ecx            # imm = 0x4ABE
	movq	%rax, -72(%rbp)         # 8-byte Spill
	movl	%ecx, -76(%rbp)         # 4-byte Spill
	je	.LBB125_3
	jmp	.LBB125_14
.LBB125_14:
	movl	-48(%rbp), %eax         # 4-byte Reload
	subl	$24601, %eax            # imm = 0x6019
	movl	%eax, -80(%rbp)         # 4-byte Spill
	je	.LBB125_4
	jmp	.LBB125_5
.LBB125_3:
	movabsq	$.L.str.29.111, %rsi
	movq	-72(%rbp), %rdi         # 8-byte Reload
	movl	-44(%rbp), %edx         # 4-byte Reload
	movq	-64(%rbp), %rcx         # 8-byte Reload
	movl	-28(%rbp), %r8d         # 4-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -84(%rbp)         # 4-byte Spill
	jmp	.LBB125_6
.LBB125_4:
	movabsq	$.L.str.30.112, %rsi
	movq	-72(%rbp), %rdi         # 8-byte Reload
	movl	-44(%rbp), %edx         # 4-byte Reload
	movq	-64(%rbp), %rcx         # 8-byte Reload
	movl	-28(%rbp), %r8d         # 4-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -88(%rbp)         # 4-byte Spill
	jmp	.LBB125_6
.LBB125_5:
	movabsq	$.L.str.31.113, %rsi
	movq	-72(%rbp), %rdi         # 8-byte Reload
	movl	-48(%rbp), %edx         # 4-byte Reload
	movl	-44(%rbp), %ecx         # 4-byte Reload
	movq	-64(%rbp), %r8          # 8-byte Reload
	movl	-28(%rbp), %r9d         # 4-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -92(%rbp)         # 4-byte Spill
.LBB125_6:
	movl	-28(%rbp), %eax         # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB125_8
# %bb.7:
	movabsq	$.L.str.22.100, %rsi
	movq	-72(%rbp), %rdi         # 8-byte Reload
	movq	-24(%rbp), %rdx         # 8-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -96(%rbp)         # 4-byte Spill
	jmp	.LBB125_9
.LBB125_8:
	movabsq	$.L.str.23.101, %rsi
	movq	-72(%rbp), %rdi         # 8-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -100(%rbp)        # 4-byte Spill
.LBB125_9:
	movl	-12(%rbp), %eax         # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB125_11
# %bb.10:
	movabsq	$.L.str.24.102, %rsi
	movq	-72(%rbp), %rdi         # 8-byte Reload
	movq	-8(%rbp), %rdx          # 8-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -104(%rbp)        # 4-byte Spill
	jmp	.LBB125_12
.LBB125_11:
	movabsq	$.L.str.25.103, %rsi
	movq	-72(%rbp), %rdi         # 8-byte Reload
	movb	$0, %al
	callq	gzprintf
	movl	%eax, -108(%rbp)        # 4-byte Spill
.LBB125_12:
	movq	-56(%rbp), %rax         # 8-byte Reload
	movq	%rax, %rsp
.LBB125_13:
	movq	%rbp, %rsp
	popq	%rbp
	retq
.Lfunc_end125:
	.size	trace_logger_log_vector, .Lfunc_end125-trace_logger_log_vector
	.cfi_endproc
                                        # -- End function
	.type	.L.str.4,@object        # @.str.4
	.section	.rodata,"a",@progbits
.L.str.4:
	.asciz	"triad"
	.size	.L.str.4, 6

	.type	.L.str.13,@object       # @.str.13
.L.str.13:
	.asciz	"a"
	.size	.L.str.13, 2

	.type	.L.str.8,@object        # @.str.8
.L.str.8:
	.asciz	"phi"
	.size	.L.str.8, 4

	.type	.L.str.16,@object       # @.str.16
.L.str.16:
	.asciz	"b"
	.size	.L.str.16, 2

	.type	.L.str.22,@object       # @.str.22
.L.str.22:
	.asciz	"c"
	.size	.L.str.22, 2

	.type	.L.str.19,@object       # @.str.19
.L.str.19:
	.asciz	"s"
	.size	.L.str.19, 2

	.type	.L.str.5,@object        # @.str.5
.L.str.5:
	.asciz	"4:0"
	.size	.L.str.5, 4

	.type	.L.str.6,@object        # @.str.6
.L.str.6:
	.asciz	"4:0-5"
	.size	.L.str.6, 6

	.type	.L.str.7,@object        # @.str.7
.L.str.7:
	.asciz	"5:1"
	.size	.L.str.7, 4

	.type	.L.str.9,@object        # @.str.9
.L.str.9:
	.asciz	"i"
	.size	.L.str.9, 2

	.type	.L.str.10,@object       # @.str.10
.L.str.10:
	.asciz	"14"
	.size	.L.str.10, 3

	.type	.L.str.11,@object       # @.str.11
.L.str.11:
	.zero	1
	.size	.L.str.11, 1

	.type	.L.str.12,@object       # @.str.12
.L.str.12:
	.asciz	"7"
	.size	.L.str.12, 2

	.type	.L.str.14,@object       # @.str.14
.L.str.14:
	.asciz	"8"
	.size	.L.str.14, 2

	.type	.L.str.15,@object       # @.str.15
.L.str.15:
	.asciz	"9"
	.size	.L.str.15, 2

	.type	.L.str.17,@object       # @.str.17
.L.str.17:
	.asciz	"10"
	.size	.L.str.17, 3

	.type	.L.str.18,@object       # @.str.18
.L.str.18:
	.asciz	"11"
	.size	.L.str.18, 3

	.type	.L.str.20,@object       # @.str.20
.L.str.20:
	.asciz	"12"
	.size	.L.str.20, 3

	.type	.L.str.21,@object       # @.str.21
.L.str.21:
	.asciz	"13"
	.size	.L.str.21, 3

	.type	.L.str.23,@object       # @.str.23
.L.str.23:
	.asciz	"5:1-1"
	.size	.L.str.23, 6

	.type	.L.str.24,@object       # @.str.24
.L.str.24:
	.asciz	"15"
	.size	.L.str.24, 3

	.type	.L.str.25,@object       # @.str.25
.L.str.25:
	.asciz	"5:1-2"
	.size	.L.str.25, 6

	.type	.L.str.26,@object       # @.str.26
.L.str.26:
	.asciz	"16:0"
	.size	.L.str.26, 5

	.type	.L.str.27,@object       # @.str.27
.L.str.27:
	.asciz	"16:0-0"
	.size	.L.str.27, 7

	.type	.L.str.28,@object       # @.str.28
.L.str.28:
	.asciz	"main"
	.size	.L.str.28, 5

	.type	.L.str.29,@object       # @.str.29
.L.str.29:
	.asciz	"0:0"
	.size	.L.str.29, 4

	.type	.L.str.30,@object       # @.str.30
.L.str.30:
	.asciz	"1"
	.size	.L.str.30, 2

	.type	.L.str.31,@object       # @.str.31
.L.str.31:
	.asciz	"malloc"
	.size	.L.str.31, 7

	.type	.L.str.32,@object       # @.str.32
.L.str.32:
	.asciz	"2"
	.size	.L.str.32, 2

	.type	.L.str.33,@object       # @.str.33
.L.str.33:
	.asciz	"3"
	.size	.L.str.33, 2

	.type	.L.str.34,@object       # @.str.34
.L.str.34:
	.asciz	"4"
	.size	.L.str.34, 2

	.type	.L.str.35,@object       # @.str.35
.L.str.35:
	.asciz	"5"
	.size	.L.str.35, 2

	.type	.L.str.36,@object       # @.str.36
.L.str.36:
	.asciz	"6"
	.size	.L.str.36, 2

	.type	.L.str.37,@object       # @.str.37
.L.str.37:
	.asciz	"time"
	.size	.L.str.37, 5

	.type	.L.str.38,@object       # @.str.38
.L.str.38:
	.asciz	"0:0-3"
	.size	.L.str.38, 6

	.type	.L.str.39,@object       # @.str.39
.L.str.39:
	.asciz	"srand"
	.size	.L.str.39, 6

	.type	.L.str.40,@object       # @.str.40
.L.str.40:
	.asciz	"0:0-5"
	.size	.L.str.40, 6

	.type	.L.str.41,@object       # @.str.41
.L.str.41:
	.asciz	"9:1"
	.size	.L.str.41, 4

	.type	.L.str.42,@object       # @.str.42
.L.str.42:
	.asciz	"16"
	.size	.L.str.42, 3

	.type	.L.str.43,@object       # @.str.43
.L.str.43:
	.asciz	"9:1-1"
	.size	.L.str.43, 6

	.type	.L.str.44,@object       # @.str.44
.L.str.44:
	.asciz	"rand"
	.size	.L.str.44, 5

	.type	.L.str.45,@object       # @.str.45
.L.str.45:
	.asciz	"9:1-2"
	.size	.L.str.45, 6

	.type	.L.str.46,@object       # @.str.46
.L.str.46:
	.asciz	"9:1-3"
	.size	.L.str.46, 6

	.type	.L.str.47,@object       # @.str.47
.L.str.47:
	.asciz	"17"
	.size	.L.str.47, 3

	.type	.L.str.48,@object       # @.str.48
.L.str.48:
	.asciz	"9:1-4"
	.size	.L.str.48, 6

	.type	.L.str.49,@object       # @.str.49
.L.str.49:
	.asciz	"18:0"
	.size	.L.str.49, 5

	.type	.L.str.50,@object       # @.str.50
.L.str.50:
	.asciz	"18:0-0"
	.size	.L.str.50, 7

	.type	.L.str.51,@object       # @.str.51
.L.str.51:
	.asciz	"output"
	.size	.L.str.51, 7

	.type	.L.str.52,@object       # @.str.52
.L.str.52:
	.asciz	"fopen"
	.size	.L.str.52, 6

	.type	.L.str.53,@object       # @.str.53
.L.str.53:
	.asciz	"output.data"
	.size	.L.str.53, 12

	.type	.L.str.54,@object       # @.str.54
.L.str.54:
	.asciz	"w"
	.size	.L.str.54, 2

	.type	.L.str,@object          # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"output.data"
	.size	.L.str, 12

	.type	.L.str.1,@object        # @.str.1
.L.str.1:
	.asciz	"w"
	.size	.L.str.1, 2

	.type	.L.str.55,@object       # @.str.55
	.section	.rodata,"a",@progbits
.L.str.55:
	.asciz	"18:0-3"
	.size	.L.str.55, 7

	.type	.L.str.56,@object       # @.str.56
.L.str.56:
	.asciz	"20:1"
	.size	.L.str.56, 5

	.type	.L.str.57,@object       # @.str.57
.L.str.57:
	.asciz	"25"
	.size	.L.str.57, 3

	.type	.L.str.58,@object       # @.str.58
.L.str.58:
	.asciz	"22"
	.size	.L.str.58, 3

	.type	.L.str.59,@object       # @.str.59
.L.str.59:
	.asciz	"23"
	.size	.L.str.59, 3

	.type	.L.str.60,@object       # @.str.60
.L.str.60:
	.asciz	"24"
	.size	.L.str.60, 3

	.type	.L.str.61,@object       # @.str.61
.L.str.61:
	.asciz	"fprintf"
	.size	.L.str.61, 8

	.type	.L.str.62,@object       # @.str.62
.L.str.62:
	.asciz	"%d,"
	.size	.L.str.62, 4

	.type	.L.str.2,@object        # @.str.2
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.2:
	.asciz	"%d,"
	.size	.L.str.2, 4

	.type	.L.str.63,@object       # @.str.63
	.section	.rodata,"a",@progbits
.L.str.63:
	.asciz	"26"
	.size	.L.str.63, 3

	.type	.L.str.64,@object       # @.str.64
.L.str.64:
	.asciz	"20:1-1"
	.size	.L.str.64, 7

	.type	.L.str.65,@object       # @.str.65
.L.str.65:
	.asciz	"27:0"
	.size	.L.str.65, 5

	.type	.L.str.66,@object       # @.str.66
.L.str.66:
	.asciz	"28"
	.size	.L.str.66, 3

	.type	.L.str.67,@object       # @.str.67
.L.str.67:
	.asciz	"\n"
	.size	.L.str.67, 2

	.type	.L.str.3,@object        # @.str.3
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.3:
	.asciz	"\n"
	.size	.L.str.3, 2

	.type	.L.str.68,@object       # @.str.68
	.section	.rodata,"a",@progbits
.L.str.68:
	.asciz	"29"
	.size	.L.str.68, 3

	.type	.L.str.69,@object       # @.str.69
.L.str.69:
	.asciz	"fclose"
	.size	.L.str.69, 7

	.type	.L.str.70,@object       # @.str.70
.L.str.70:
	.asciz	"27:0-0"
	.size	.L.str.70, 7

	.type	trace,@object           # @trace
	.section	.tbss,"awT",@nobits
	.globl	trace
	.p2align	3
trace:
	.quad	0
	.size	trace, 8

	.type	_Z8gz_filesB5cxx11,@object # @_Z8gz_filesB5cxx11
	.bss
	.globl	_Z8gz_filesB5cxx11
	.p2align	3
_Z8gz_filesB5cxx11:
	.zero	48
	.size	_Z8gz_filesB5cxx11, 48

	.type	lock,@object            # @lock
	.globl	lock
	.p2align	3
lock:
	.zero	40
	.size	lock, 40

	.type	_Z12labelmap_strB5cxx11,@object # @_Z12labelmap_strB5cxx11
	.globl	_Z12labelmap_strB5cxx11
	.p2align	3
_Z12labelmap_strB5cxx11:
	.zero	32
	.size	_Z12labelmap_strB5cxx11, 32

	.type	default_trace_name,@object # @default_trace_name
	.data
	.globl	default_trace_name
	.p2align	3
default_trace_name:
	.quad	.L.str.71
	.size	default_trace_name, 8

	.type	.L.str.71,@object       # @.str.71
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.71:
	.asciz	"dynamic_trace.gz"
	.size	.L.str.71, 17

	.section	.init_array,"aw",@init_array
	.p2align	3
	.quad	_GLOBAL__sub_I_trace_logger.cpp
	.type	.L.str.3.72,@object     # @.str.3.72
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.3.72:
	.asciz	"!trace && \"Trace has already been created!\""
	.size	.L.str.3.72, 44

	.type	.L.str.4.73,@object     # @.str.4.73
.L.str.4.73:
	.asciz	"/workspace/LLVM-Tracer/profile-func/trace_logger.cpp"
	.size	.L.str.4.73, 53

	.type	.L__PRETTY_FUNCTION__._Z12create_tracePKc,@object # @__PRETTY_FUNCTION__._Z12create_tracePKc
.L__PRETTY_FUNCTION__._Z12create_tracePKc:
	.asciz	"void create_trace(const char *)"
	.size	.L__PRETTY_FUNCTION__._Z12create_tracePKc, 32

	.type	.L.str.5.74,@object     # @.str.5.74
.L.str.5.74:
	.asciz	"%%%% LABEL MAP START %%%%\n"
	.size	.L.str.5.74, 27

	.type	.L.str.6.75,@object     # @.str.6.75
.L.str.6.75:
	.asciz	"%%%% LABEL MAP END %%%%\n\n"
	.size	.L.str.6.75, 26

	.type	.L.str.7.76,@object     # @.str.7.76
.L.str.7.76:
	.asciz	"w"
	.size	.L.str.7.76, 2

	.type	.L.str.8.77,@object     # @.str.8.77
.L.str.8.77:
	.asciz	"Failed to open logfile \"dynamic_trace\""
	.size	.L.str.8.77, 39

	.type	_ZStL19piecewise_construct,@object # @_ZStL19piecewise_construct
	.section	.rodata,"a",@progbits
_ZStL19piecewise_construct:
	.zero	1
	.size	_ZStL19piecewise_construct, 1

	.type	.L.str.36.78,@object    # @.str.36.78
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.36.78:
	.asciz	"map::at"
	.size	.L.str.36.78, 8

	.type	.L.str.9.79,@object     # @.str.9.79
.L.str.9.79:
	.asciz	"Failed to initialize the mutex\n"
	.size	.L.str.9.79, 32

	.type	.L.str.10.80,@object    # @.str.10.80
.L.str.10.80:
	.asciz	"false && \"Returning from within a toplevel function before it was called!\""
	.size	.L.str.10.80, 75

	.type	.L__PRETTY_FUNCTION__._Z10log_or_notbbiPc,@object # @__PRETTY_FUNCTION__._Z10log_or_notbbiPc
.L__PRETTY_FUNCTION__._Z10log_or_notbbiPc:
	.asciz	"logging_status log_or_not(bool, bool, int, char *)"
	.size	.L__PRETTY_FUNCTION__._Z10log_or_notbbiPc, 51

	.type	.L.str.11.81,@object    # @.str.11.81
.L.str.11.81:
	.asciz	"false && \"Cannot call a top level function from within another one!\""
	.size	.L.str.11.81, 69

	.type	.L.str.12.82,@object    # @.str.12.82
.L.str.12.82:
	.asciz	"0x"
	.size	.L.str.12.82, 3

	.type	.L.str.13.83,@object    # @.str.13.83
.L.str.13.83:
	.asciz	"%02x"
	.size	.L.str.13.83, 5

	.type	.L.str.14.86,@object    # @.str.14.86
.L.str.14.86:
	.asciz	"%s: Stopping logging at inst %ld.\n"
	.size	.L.str.14.86, 35

	.type	.L.str.15.87,@object    # @.str.15.87
.L.str.15.87:
	.asciz	"%s: Starting to log at inst = %ld.\n"
	.size	.L.str.15.87, 36

	.type	.L.str.16.88,@object    # @.str.16.88
.L.str.16.88:
	.zero	1
	.size	.L.str.16.88, 1

	.type	.L.str.17.91,@object    # @.str.17.91
.L.str.17.91:
	.asciz	"\nentry,%s,%d,\n"
	.size	.L.str.17.91, 15

	.type	.L.str.18.94,@object    # @.str.18.94
.L.str.18.94:
	.asciz	"\n0,%d,%s,%s,%s,%d,%ld\n"
	.size	.L.str.18.94, 23

	.type	.L.str.19.97,@object    # @.str.19.97
.L.str.19.97:
	.asciz	"r,%d,%ld,%d"
	.size	.L.str.19.97, 12

	.type	.L.str.20.98,@object    # @.str.20.98
.L.str.20.98:
	.asciz	"f,%d,%ld,%d"
	.size	.L.str.20.98, 12

	.type	.L.str.21.99,@object    # @.str.21.99
.L.str.21.99:
	.asciz	"%d,%d,%ld,%d"
	.size	.L.str.21.99, 13

	.type	.L.str.22.100,@object   # @.str.22.100
.L.str.22.100:
	.asciz	",%s"
	.size	.L.str.22.100, 4

	.type	.L.str.23.101,@object   # @.str.23.101
.L.str.23.101:
	.asciz	", "
	.size	.L.str.23.101, 3

	.type	.L.str.24.102,@object   # @.str.24.102
.L.str.24.102:
	.asciz	",%s,\n"
	.size	.L.str.24.102, 6

	.type	.L.str.25.103,@object   # @.str.25.103
.L.str.25.103:
	.asciz	",\n"
	.size	.L.str.25.103, 3

	.type	.L.str.26.106,@object   # @.str.26.106
.L.str.26.106:
	.asciz	"r,%d,%#llx,%d"
	.size	.L.str.26.106, 14

	.type	.L.str.27.107,@object   # @.str.27.107
.L.str.27.107:
	.asciz	"f,%d,%#llx,%d"
	.size	.L.str.27.107, 14

	.type	.L.str.28.108,@object   # @.str.28.108
.L.str.28.108:
	.asciz	"%d,%d,%#llx,%d"
	.size	.L.str.28.108, 15

	.type	.L.str.29.111,@object   # @.str.29.111
.L.str.29.111:
	.asciz	"r,%d,%s,%d"
	.size	.L.str.29.111, 11

	.type	.L.str.30.112,@object   # @.str.30.112
.L.str.30.112:
	.asciz	"f,%d,%s,%d"
	.size	.L.str.30.112, 11

	.type	.L.str.31.113,@object   # @.str.31.113
.L.str.31.113:
	.asciz	"%d,%d,%s,%d"
	.size	.L.str.31.113, 12

	.type	.L.str.32.114,@object   # @.str.32.114
.L.str.32.114:
	.asciz	"r,%d,%f,%d"
	.size	.L.str.32.114, 11

	.type	.L.str.33.115,@object   # @.str.33.115
.L.str.33.115:
	.asciz	"f,%d,%f,%d"
	.size	.L.str.33.115, 11

	.type	.L.str.34.116,@object   # @.str.34.116
.L.str.34.116:
	.asciz	"%d,%d,%f,%d"
	.size	.L.str.34.116, 12

	.file	2 "/usr/include/x86_64-linux-gnu/bits/libio.h"
	.file	3 "/usr/include/x86_64-linux-gnu/bits/types.h"
	.file	4 "/usr/local/lib/clang/6.0.0/include/stddef.h"
	.file	5 "/usr/include/x86_64-linux-gnu/bits/types/FILE.h"
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 6.0.0 (tags/RELEASE_600/final)" # string offset=0
.Linfo_string1:
	.asciz	"/workspace/LLVM-Tracer/example/triad/triad.c" # string offset=45
.Linfo_string2:
	.asciz	"/workspace/LLVM-Tracer/build/example/triad" # string offset=90
.Linfo_string3:
	.asciz	"int"                   # string offset=133
.Linfo_string4:
	.asciz	"triad"                 # string offset=137
.Linfo_string5:
	.asciz	"main"                  # string offset=143
.Linfo_string6:
	.asciz	"a"                     # string offset=148
.Linfo_string7:
	.asciz	"b"                     # string offset=150
.Linfo_string8:
	.asciz	"c"                     # string offset=152
.Linfo_string9:
	.asciz	"s"                     # string offset=154
.Linfo_string10:
	.asciz	"i"                     # string offset=156
.Linfo_string11:
	.asciz	"output"                # string offset=158
.Linfo_string12:
	.asciz	"_flags"                # string offset=165
.Linfo_string13:
	.asciz	"_IO_read_ptr"          # string offset=172
.Linfo_string14:
	.asciz	"char"                  # string offset=185
.Linfo_string15:
	.asciz	"_IO_read_end"          # string offset=190
.Linfo_string16:
	.asciz	"_IO_read_base"         # string offset=203
.Linfo_string17:
	.asciz	"_IO_write_base"        # string offset=217
.Linfo_string18:
	.asciz	"_IO_write_ptr"         # string offset=232
.Linfo_string19:
	.asciz	"_IO_write_end"         # string offset=246
.Linfo_string20:
	.asciz	"_IO_buf_base"          # string offset=260
.Linfo_string21:
	.asciz	"_IO_buf_end"           # string offset=273
.Linfo_string22:
	.asciz	"_IO_save_base"         # string offset=285
.Linfo_string23:
	.asciz	"_IO_backup_base"       # string offset=299
.Linfo_string24:
	.asciz	"_IO_save_end"          # string offset=315
.Linfo_string25:
	.asciz	"_markers"              # string offset=328
.Linfo_string26:
	.asciz	"_next"                 # string offset=337
.Linfo_string27:
	.asciz	"_sbuf"                 # string offset=343
.Linfo_string28:
	.asciz	"_pos"                  # string offset=349
.Linfo_string29:
	.asciz	"_IO_marker"            # string offset=354
.Linfo_string30:
	.asciz	"_chain"                # string offset=365
.Linfo_string31:
	.asciz	"_fileno"               # string offset=372
.Linfo_string32:
	.asciz	"_flags2"               # string offset=380
.Linfo_string33:
	.asciz	"_old_offset"           # string offset=388
.Linfo_string34:
	.asciz	"long int"              # string offset=400
.Linfo_string35:
	.asciz	"__off_t"               # string offset=409
.Linfo_string36:
	.asciz	"_cur_column"           # string offset=417
.Linfo_string37:
	.asciz	"unsigned short"        # string offset=429
.Linfo_string38:
	.asciz	"_vtable_offset"        # string offset=444
.Linfo_string39:
	.asciz	"signed char"           # string offset=459
.Linfo_string40:
	.asciz	"_shortbuf"             # string offset=471
.Linfo_string41:
	.asciz	"sizetype"              # string offset=481
.Linfo_string42:
	.asciz	"_lock"                 # string offset=490
.Linfo_string43:
	.asciz	"_IO_lock_t"            # string offset=496
.Linfo_string44:
	.asciz	"_offset"               # string offset=507
.Linfo_string45:
	.asciz	"__off64_t"             # string offset=515
.Linfo_string46:
	.asciz	"__pad1"                # string offset=525
.Linfo_string47:
	.asciz	"__pad2"                # string offset=532
.Linfo_string48:
	.asciz	"__pad3"                # string offset=539
.Linfo_string49:
	.asciz	"__pad4"                # string offset=546
.Linfo_string50:
	.asciz	"__pad5"                # string offset=553
.Linfo_string51:
	.asciz	"long unsigned int"     # string offset=560
.Linfo_string52:
	.asciz	"size_t"                # string offset=578
.Linfo_string53:
	.asciz	"_mode"                 # string offset=585
.Linfo_string54:
	.asciz	"_unused2"              # string offset=591
.Linfo_string55:
	.asciz	"_IO_FILE"              # string offset=600
.Linfo_string56:
	.asciz	"FILE"                  # string offset=609
	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
	.quad	.Lfunc_begin0-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	2                       # Loc expr size
	.byte	118                     # DW_OP_breg6
	.byte	88                      # -40
	.quad	0
	.quad	0
.Ldebug_loc1:
	.quad	.Lfunc_begin0-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	2                       # Loc expr size
	.byte	118                     # DW_OP_breg6
	.byte	80                      # -48
	.quad	0
	.quad	0
.Ldebug_loc2:
	.quad	.Lfunc_begin0-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	2                       # Loc expr size
	.byte	118                     # DW_OP_breg6
	.byte	72                      # -56
	.quad	0
	.quad	0
.Ldebug_loc3:
	.quad	.Lfunc_begin0-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	2                       # Loc expr size
	.byte	118                     # DW_OP_breg6
	.byte	68                      # -60
	.quad	0
	.quad	0
.Ldebug_loc4:
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	17                      # DW_OP_consts
	.byte	0                       # 0
	.byte	159                     # DW_OP_stack_value
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	118                     # DW_OP_breg6
	.byte	152                     # -104
	.byte	127                     # 
	.quad	0
	.quad	0
.Ldebug_loc5:
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Lfunc_end1-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	118                     # DW_OP_breg6
	.byte	160                     # -96
	.byte	127                     # 
	.quad	0
	.quad	0
.Ldebug_loc6:
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Lfunc_end1-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	118                     # DW_OP_breg6
	.byte	232                     # -152
	.byte	126                     # 
	.quad	0
	.quad	0
.Ldebug_loc7:
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Lfunc_end1-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	118                     # DW_OP_breg6
	.byte	176                     # -208
	.byte	126                     # 
	.quad	0
	.quad	0
.Ldebug_loc8:
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	17                      # DW_OP_consts
	.byte	0                       # 0
	.byte	159                     # DW_OP_stack_value
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	118                     # DW_OP_breg6
	.byte	200                     # -312
	.byte	125                     # 
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	17                      # DW_OP_consts
	.byte	0                       # 0
	.byte	159                     # DW_OP_stack_value
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	118                     # DW_OP_breg6
	.byte	160                     # -864
	.byte	121                     # 
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Lfunc_end1-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	118                     # DW_OP_breg6
	.byte	200                     # -312
	.byte	125                     # 
	.quad	0
	.quad	0
.Ldebug_loc9:
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Lfunc_end1-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	118                     # DW_OP_breg6
	.byte	200                     # -824
	.byte	121                     # 
	.quad	0
	.quad	0
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	37                      # DW_AT_producer
	.byte	14                      # DW_FORM_strp
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	16                      # DW_AT_stmt_list
	.byte	23                      # DW_FORM_sec_offset
	.byte	27                      # DW_AT_comp_dir
	.byte	14                      # DW_FORM_strp
	.ascii	"\264B"                 # DW_AT_GNU_pubnames
	.byte	25                      # DW_FORM_flag_present
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	15                      # DW_TAG_pointer_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	3                       # Abbreviation Code
	.byte	36                      # DW_TAG_base_type
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	62                      # DW_AT_encoding
	.byte	11                      # DW_FORM_data1
	.byte	11                      # DW_AT_byte_size
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	4                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	64                      # DW_AT_frame_base
	.byte	24                      # DW_FORM_exprloc
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	39                      # DW_AT_prototyped
	.byte	25                      # DW_FORM_flag_present
	.byte	63                      # DW_AT_external
	.byte	25                      # DW_FORM_flag_present
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	5                       # Abbreviation Code
	.byte	5                       # DW_TAG_formal_parameter
	.byte	0                       # DW_CHILDREN_no
	.byte	2                       # DW_AT_location
	.byte	23                      # DW_FORM_sec_offset
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	6                       # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	2                       # DW_AT_location
	.byte	23                      # DW_FORM_sec_offset
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	7                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	64                      # DW_AT_frame_base
	.byte	24                      # DW_FORM_exprloc
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	63                      # DW_AT_external
	.byte	25                      # DW_FORM_flag_present
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	8                       # Abbreviation Code
	.byte	22                      # DW_TAG_typedef
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	9                       # Abbreviation Code
	.byte	19                      # DW_TAG_structure_type
	.byte	1                       # DW_CHILDREN_yes
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	11                      # DW_AT_byte_size
	.byte	11                      # DW_FORM_data1
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	10                      # Abbreviation Code
	.byte	13                      # DW_TAG_member
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	56                      # DW_AT_data_member_location
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	11                      # Abbreviation Code
	.byte	13                      # DW_TAG_member
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	5                       # DW_FORM_data2
	.byte	56                      # DW_AT_data_member_location
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	12                      # Abbreviation Code
	.byte	1                       # DW_TAG_array_type
	.byte	1                       # DW_CHILDREN_yes
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	13                      # Abbreviation Code
	.byte	33                      # DW_TAG_subrange_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	55                      # DW_AT_count
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	14                      # Abbreviation Code
	.byte	36                      # DW_TAG_base_type
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	11                      # DW_AT_byte_size
	.byte	11                      # DW_FORM_data1
	.byte	62                      # DW_AT_encoding
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	15                      # Abbreviation Code
	.byte	22                      # DW_TAG_typedef
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	16                      # Abbreviation Code
	.byte	15                      # DW_TAG_pointer_type
	.byte	0                       # DW_CHILDREN_no
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	817                     # Length of Unit
	.short	4                       # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] 0xb:0x32a DW_TAG_compile_unit
	.long	.Linfo_string0          # DW_AT_producer
	.short	12                      # DW_AT_language
	.long	.Linfo_string1          # DW_AT_name
	.long	.Lline_table_start0     # DW_AT_stmt_list
	.long	.Linfo_string2          # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin0 # DW_AT_high_pc
	.byte	2                       # Abbrev [2] 0x2a:0x5 DW_TAG_pointer_type
	.long	47                      # DW_AT_type
	.byte	3                       # Abbrev [3] 0x2f:0x7 DW_TAG_base_type
	.long	.Linfo_string3          # DW_AT_name
	.byte	5                       # DW_AT_encoding
	.byte	4                       # DW_AT_byte_size
	.byte	4                       # Abbrev [4] 0x36:0x61 DW_TAG_subprogram
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
	.byte	1                       # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string4          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	3                       # DW_AT_decl_line
                                        # DW_AT_prototyped
                                        # DW_AT_external
	.byte	5                       # Abbrev [5] 0x4b:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc0            # DW_AT_location
	.long	.Linfo_string6          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	3                       # DW_AT_decl_line
	.long	42                      # DW_AT_type
	.byte	5                       # Abbrev [5] 0x5a:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc1            # DW_AT_location
	.long	.Linfo_string7          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	3                       # DW_AT_decl_line
	.long	42                      # DW_AT_type
	.byte	5                       # Abbrev [5] 0x69:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc2            # DW_AT_location
	.long	.Linfo_string8          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	3                       # DW_AT_decl_line
	.long	42                      # DW_AT_type
	.byte	5                       # Abbrev [5] 0x78:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc3            # DW_AT_location
	.long	.Linfo_string9          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	3                       # DW_AT_decl_line
	.long	47                      # DW_AT_type
	.byte	6                       # Abbrev [6] 0x87:0xf DW_TAG_variable
	.long	.Ldebug_loc4            # DW_AT_location
	.long	.Linfo_string10         # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	4                       # DW_AT_decl_line
	.long	47                      # DW_AT_type
	.byte	0                       # End Of Children Mark
	.byte	7                       # Abbrev [7] 0x97:0x65 DW_TAG_subprogram
	.quad	.Lfunc_begin1           # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1 # DW_AT_high_pc
	.byte	1                       # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string5          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	10                      # DW_AT_decl_line
	.long	47                      # DW_AT_type
                                        # DW_AT_external
	.byte	6                       # Abbrev [6] 0xb0:0xf DW_TAG_variable
	.long	.Ldebug_loc5            # DW_AT_location
	.long	.Linfo_string6          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	11                      # DW_AT_decl_line
	.long	42                      # DW_AT_type
	.byte	6                       # Abbrev [6] 0xbf:0xf DW_TAG_variable
	.long	.Ldebug_loc6            # DW_AT_location
	.long	.Linfo_string7          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	11                      # DW_AT_decl_line
	.long	42                      # DW_AT_type
	.byte	6                       # Abbrev [6] 0xce:0xf DW_TAG_variable
	.long	.Ldebug_loc7            # DW_AT_location
	.long	.Linfo_string8          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	11                      # DW_AT_decl_line
	.long	42                      # DW_AT_type
	.byte	6                       # Abbrev [6] 0xdd:0xf DW_TAG_variable
	.long	.Ldebug_loc8            # DW_AT_location
	.long	.Linfo_string10         # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	15                      # DW_AT_decl_line
	.long	47                      # DW_AT_type
	.byte	6                       # Abbrev [6] 0xec:0xf DW_TAG_variable
	.long	.Ldebug_loc9            # DW_AT_location
	.long	.Linfo_string11         # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	25                      # DW_AT_decl_line
	.long	252                     # DW_AT_type
	.byte	0                       # End Of Children Mark
	.byte	2                       # Abbrev [2] 0xfc:0x5 DW_TAG_pointer_type
	.long	257                     # DW_AT_type
	.byte	8                       # Abbrev [8] 0x101:0xb DW_TAG_typedef
	.long	268                     # DW_AT_type
	.long	.Linfo_string56         # DW_AT_name
	.byte	5                       # DW_AT_decl_file
	.byte	7                       # DW_AT_decl_line
	.byte	9                       # Abbrev [9] 0x10c:0x17c DW_TAG_structure_type
	.long	.Linfo_string55         # DW_AT_name
	.byte	216                     # DW_AT_byte_size
	.byte	2                       # DW_AT_decl_file
	.byte	245                     # DW_AT_decl_line
	.byte	10                      # Abbrev [10] 0x114:0xc DW_TAG_member
	.long	.Linfo_string12         # DW_AT_name
	.long	47                      # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.byte	246                     # DW_AT_decl_line
	.byte	0                       # DW_AT_data_member_location
	.byte	10                      # Abbrev [10] 0x120:0xc DW_TAG_member
	.long	.Linfo_string13         # DW_AT_name
	.long	648                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.byte	251                     # DW_AT_decl_line
	.byte	8                       # DW_AT_data_member_location
	.byte	10                      # Abbrev [10] 0x12c:0xc DW_TAG_member
	.long	.Linfo_string15         # DW_AT_name
	.long	648                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.byte	252                     # DW_AT_decl_line
	.byte	16                      # DW_AT_data_member_location
	.byte	10                      # Abbrev [10] 0x138:0xc DW_TAG_member
	.long	.Linfo_string16         # DW_AT_name
	.long	648                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.byte	253                     # DW_AT_decl_line
	.byte	24                      # DW_AT_data_member_location
	.byte	10                      # Abbrev [10] 0x144:0xc DW_TAG_member
	.long	.Linfo_string17         # DW_AT_name
	.long	648                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.byte	254                     # DW_AT_decl_line
	.byte	32                      # DW_AT_data_member_location
	.byte	10                      # Abbrev [10] 0x150:0xc DW_TAG_member
	.long	.Linfo_string18         # DW_AT_name
	.long	648                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.byte	255                     # DW_AT_decl_line
	.byte	40                      # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x15c:0xd DW_TAG_member
	.long	.Linfo_string19         # DW_AT_name
	.long	648                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	256                     # DW_AT_decl_line
	.byte	48                      # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x169:0xd DW_TAG_member
	.long	.Linfo_string20         # DW_AT_name
	.long	648                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	257                     # DW_AT_decl_line
	.byte	56                      # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x176:0xd DW_TAG_member
	.long	.Linfo_string21         # DW_AT_name
	.long	648                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	258                     # DW_AT_decl_line
	.byte	64                      # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x183:0xd DW_TAG_member
	.long	.Linfo_string22         # DW_AT_name
	.long	648                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	260                     # DW_AT_decl_line
	.byte	72                      # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x190:0xd DW_TAG_member
	.long	.Linfo_string23         # DW_AT_name
	.long	648                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	261                     # DW_AT_decl_line
	.byte	80                      # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x19d:0xd DW_TAG_member
	.long	.Linfo_string24         # DW_AT_name
	.long	648                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	262                     # DW_AT_decl_line
	.byte	88                      # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x1aa:0xd DW_TAG_member
	.long	.Linfo_string25         # DW_AT_name
	.long	660                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	264                     # DW_AT_decl_line
	.byte	96                      # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x1b7:0xd DW_TAG_member
	.long	.Linfo_string30         # DW_AT_name
	.long	710                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	266                     # DW_AT_decl_line
	.byte	104                     # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x1c4:0xd DW_TAG_member
	.long	.Linfo_string31         # DW_AT_name
	.long	47                      # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	268                     # DW_AT_decl_line
	.byte	112                     # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x1d1:0xd DW_TAG_member
	.long	.Linfo_string32         # DW_AT_name
	.long	47                      # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	272                     # DW_AT_decl_line
	.byte	116                     # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x1de:0xd DW_TAG_member
	.long	.Linfo_string33         # DW_AT_name
	.long	715                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	274                     # DW_AT_decl_line
	.byte	120                     # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x1eb:0xd DW_TAG_member
	.long	.Linfo_string36         # DW_AT_name
	.long	733                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	278                     # DW_AT_decl_line
	.byte	128                     # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x1f8:0xd DW_TAG_member
	.long	.Linfo_string38         # DW_AT_name
	.long	740                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	279                     # DW_AT_decl_line
	.byte	130                     # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x205:0xd DW_TAG_member
	.long	.Linfo_string40         # DW_AT_name
	.long	747                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	280                     # DW_AT_decl_line
	.byte	131                     # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x212:0xd DW_TAG_member
	.long	.Linfo_string42         # DW_AT_name
	.long	766                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	284                     # DW_AT_decl_line
	.byte	136                     # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x21f:0xd DW_TAG_member
	.long	.Linfo_string44         # DW_AT_name
	.long	778                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	293                     # DW_AT_decl_line
	.byte	144                     # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x22c:0xd DW_TAG_member
	.long	.Linfo_string46         # DW_AT_name
	.long	789                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	301                     # DW_AT_decl_line
	.byte	152                     # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x239:0xd DW_TAG_member
	.long	.Linfo_string47         # DW_AT_name
	.long	789                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	302                     # DW_AT_decl_line
	.byte	160                     # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x246:0xd DW_TAG_member
	.long	.Linfo_string48         # DW_AT_name
	.long	789                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	303                     # DW_AT_decl_line
	.byte	168                     # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x253:0xd DW_TAG_member
	.long	.Linfo_string49         # DW_AT_name
	.long	789                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	304                     # DW_AT_decl_line
	.byte	176                     # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x260:0xd DW_TAG_member
	.long	.Linfo_string50         # DW_AT_name
	.long	790                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	306                     # DW_AT_decl_line
	.byte	184                     # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x26d:0xd DW_TAG_member
	.long	.Linfo_string53         # DW_AT_name
	.long	47                      # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	307                     # DW_AT_decl_line
	.byte	192                     # DW_AT_data_member_location
	.byte	11                      # Abbrev [11] 0x27a:0xd DW_TAG_member
	.long	.Linfo_string54         # DW_AT_name
	.long	808                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.short	309                     # DW_AT_decl_line
	.byte	196                     # DW_AT_data_member_location
	.byte	0                       # End Of Children Mark
	.byte	2                       # Abbrev [2] 0x288:0x5 DW_TAG_pointer_type
	.long	653                     # DW_AT_type
	.byte	3                       # Abbrev [3] 0x28d:0x7 DW_TAG_base_type
	.long	.Linfo_string14         # DW_AT_name
	.byte	6                       # DW_AT_encoding
	.byte	1                       # DW_AT_byte_size
	.byte	2                       # Abbrev [2] 0x294:0x5 DW_TAG_pointer_type
	.long	665                     # DW_AT_type
	.byte	9                       # Abbrev [9] 0x299:0x2d DW_TAG_structure_type
	.long	.Linfo_string29         # DW_AT_name
	.byte	24                      # DW_AT_byte_size
	.byte	2                       # DW_AT_decl_file
	.byte	160                     # DW_AT_decl_line
	.byte	10                      # Abbrev [10] 0x2a1:0xc DW_TAG_member
	.long	.Linfo_string26         # DW_AT_name
	.long	660                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.byte	161                     # DW_AT_decl_line
	.byte	0                       # DW_AT_data_member_location
	.byte	10                      # Abbrev [10] 0x2ad:0xc DW_TAG_member
	.long	.Linfo_string27         # DW_AT_name
	.long	710                     # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.byte	162                     # DW_AT_decl_line
	.byte	8                       # DW_AT_data_member_location
	.byte	10                      # Abbrev [10] 0x2b9:0xc DW_TAG_member
	.long	.Linfo_string28         # DW_AT_name
	.long	47                      # DW_AT_type
	.byte	2                       # DW_AT_decl_file
	.byte	166                     # DW_AT_decl_line
	.byte	16                      # DW_AT_data_member_location
	.byte	0                       # End Of Children Mark
	.byte	2                       # Abbrev [2] 0x2c6:0x5 DW_TAG_pointer_type
	.long	268                     # DW_AT_type
	.byte	8                       # Abbrev [8] 0x2cb:0xb DW_TAG_typedef
	.long	726                     # DW_AT_type
	.long	.Linfo_string35         # DW_AT_name
	.byte	3                       # DW_AT_decl_file
	.byte	140                     # DW_AT_decl_line
	.byte	3                       # Abbrev [3] 0x2d6:0x7 DW_TAG_base_type
	.long	.Linfo_string34         # DW_AT_name
	.byte	5                       # DW_AT_encoding
	.byte	8                       # DW_AT_byte_size
	.byte	3                       # Abbrev [3] 0x2dd:0x7 DW_TAG_base_type
	.long	.Linfo_string37         # DW_AT_name
	.byte	7                       # DW_AT_encoding
	.byte	2                       # DW_AT_byte_size
	.byte	3                       # Abbrev [3] 0x2e4:0x7 DW_TAG_base_type
	.long	.Linfo_string39         # DW_AT_name
	.byte	6                       # DW_AT_encoding
	.byte	1                       # DW_AT_byte_size
	.byte	12                      # Abbrev [12] 0x2eb:0xc DW_TAG_array_type
	.long	653                     # DW_AT_type
	.byte	13                      # Abbrev [13] 0x2f0:0x6 DW_TAG_subrange_type
	.long	759                     # DW_AT_type
	.byte	1                       # DW_AT_count
	.byte	0                       # End Of Children Mark
	.byte	14                      # Abbrev [14] 0x2f7:0x7 DW_TAG_base_type
	.long	.Linfo_string41         # DW_AT_name
	.byte	8                       # DW_AT_byte_size
	.byte	7                       # DW_AT_encoding
	.byte	2                       # Abbrev [2] 0x2fe:0x5 DW_TAG_pointer_type
	.long	771                     # DW_AT_type
	.byte	15                      # Abbrev [15] 0x303:0x7 DW_TAG_typedef
	.long	.Linfo_string43         # DW_AT_name
	.byte	2                       # DW_AT_decl_file
	.byte	154                     # DW_AT_decl_line
	.byte	8                       # Abbrev [8] 0x30a:0xb DW_TAG_typedef
	.long	726                     # DW_AT_type
	.long	.Linfo_string45         # DW_AT_name
	.byte	3                       # DW_AT_decl_file
	.byte	141                     # DW_AT_decl_line
	.byte	16                      # Abbrev [16] 0x315:0x1 DW_TAG_pointer_type
	.byte	8                       # Abbrev [8] 0x316:0xb DW_TAG_typedef
	.long	801                     # DW_AT_type
	.long	.Linfo_string52         # DW_AT_name
	.byte	4                       # DW_AT_decl_file
	.byte	62                      # DW_AT_decl_line
	.byte	3                       # Abbrev [3] 0x321:0x7 DW_TAG_base_type
	.long	.Linfo_string51         # DW_AT_name
	.byte	7                       # DW_AT_encoding
	.byte	8                       # DW_AT_byte_size
	.byte	12                      # Abbrev [12] 0x328:0xc DW_TAG_array_type
	.long	653                     # DW_AT_type
	.byte	13                      # Abbrev [13] 0x32d:0x6 DW_TAG_subrange_type
	.long	759                     # DW_AT_type
	.byte	20                      # DW_AT_count
	.byte	0                       # End Of Children Mark
	.byte	0                       # End Of Children Mark
	.section	.debug_ranges,"",@progbits
	.section	.debug_macinfo,"",@progbits
.Lcu_macro_begin0:
	.byte	0                       # End Of Macro List Mark
	.section	.debug_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_begin0 # Length of Public Names Info
.LpubNames_begin0:
	.short	2                       # DWARF Version
	.long	.Lcu_begin0             # Offset of Compilation Unit Info
	.long	821                     # Compilation Unit Length
	.long	54                      # DIE offset
	.asciz	"triad"                 # External Name
	.long	151                     # DIE offset
	.asciz	"main"                  # External Name
	.long	0                       # End Mark
.LpubNames_end0:
	.section	.debug_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_begin0 # Length of Public Types Info
.LpubTypes_begin0:
	.short	2                       # DWARF Version
	.long	.Lcu_begin0             # Offset of Compilation Unit Info
	.long	821                     # Compilation Unit Length
	.long	257                     # DIE offset
	.asciz	"FILE"                  # External Name
	.long	47                      # DIE offset
	.asciz	"int"                   # External Name
	.long	715                     # DIE offset
	.asciz	"__off_t"               # External Name
	.long	733                     # DIE offset
	.asciz	"unsigned short"        # External Name
	.long	790                     # DIE offset
	.asciz	"size_t"                # External Name
	.long	771                     # DIE offset
	.asciz	"_IO_lock_t"            # External Name
	.long	268                     # DIE offset
	.asciz	"_IO_FILE"              # External Name
	.long	778                     # DIE offset
	.asciz	"__off64_t"             # External Name
	.long	801                     # DIE offset
	.asciz	"long unsigned int"     # External Name
	.long	665                     # DIE offset
	.asciz	"_IO_marker"            # External Name
	.long	726                     # DIE offset
	.asciz	"long int"              # External Name
	.long	740                     # DIE offset
	.asciz	"signed char"           # External Name
	.long	653                     # DIE offset
	.asciz	"char"                  # External Name
	.long	0                       # End Mark
.LpubTypes_end0:

	.ident	"clang version 6.0.0 (tags/RELEASE_600/final)"
	.ident	"clang version 6.0.0 (tags/RELEASE_600/final)"
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
