#include "common/vci-msg.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <signal.h>

#include <unistd.h>

#include <sys/un.h>
#include <sys/socket.h>

#include <dpu.h>
#include <dpu_debug.h>
#include <dpu_runner.h>
#include <dpu_memory.h>
#include <dpu_management.h>
#include <dpu_transfer_matrix.h>

typedef struct cli_args {
    int nr_ranks;
} cli_args;

typedef struct msg_queue {
    int sock;
    struct sockaddr_un resp_addr;
    socklen_t resp_len;
} msg_queue;

typedef struct switch_state {
    struct {
        struct dpu_rank_t* rank;
    } ranks[40];
} switch_state;

static const char* s_dpu_profile = "backend=hw,rankMode=perf";

/** set to true on SIGTERM and SIGINT to properly cleanup allocated data before exiting */
volatile static bool s_sig_term_received = false;

static void on_sig_received(__attribute__((unused)) int n) {
    if (!s_sig_term_received) {
        s_sig_term_received = true;
    } else {
        printf("[EXIT] Received signal multiple times: Exiting\n");
        exit(EXIT_FAILURE);
    }
}

__attribute__((constructor))
static void setup_signal_handlers(void) {
    signal(SIGTERM, on_sig_received);
    signal(SIGINT, on_sig_received);
}

/** print out message in readable format */
static void log_ci_msg(const char* prefix, const char* param, vci_msg msg) {
    char buf[120];
    vci_msg_to_string(msg, buf);

    if (param) {
        printf("[%s %s] %s\n", prefix, param, buf);
    } else {
        printf("[%s] %s\n", prefix, buf);
    }
}

/** create a unix socket receiving abstract ci commands */
static msg_queue init_unix_socket(void) {
    int sock = -1;
    struct sockaddr_un addr = { 0 };

    if (unlink(VCI_SOCKET_NAME) < 0 && errno != ENOENT) {
        goto error;
    }

    if ((sock = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
        goto error;
    }

    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, VCI_SOCKET_NAME);

    if (bind(sock, (struct sockaddr*) &addr, sizeof(addr)) < 0) {
        goto error;
    }

    return (msg_queue) { .sock = sock };

error:
    if (sock) {
        close(sock);
    }

    printf("[FAIL] cannot create socket: %s\n", strerror(errno));
    return (msg_queue) { .sock = -1 };
}

/** receive and validate message from the socket */
static vci_msg recv_ci_msg(msg_queue* q) {
    vci_msg res = { 0 };
    q->resp_len = sizeof(struct sockaddr_un);

    if (recvfrom(q->sock, &res, sizeof(res), 0, (void*) &q->resp_addr, &q->resp_len) != sizeof(res)) {
        res.type = VCI_SYS_ERR;
        return res;
    }

    if (res.type < 0 || res.type == VCI_STATUS || res.type == VCI_OK || res.type > VCI_RST_DPUS + 1 || res.ci_nr >= 8) {
        res.type = VCI_MSG_ERR;
    }

    log_ci_msg("RECV", q->resp_addr.sun_path, res);
    return res;
}

/** validate and sent a message to the socket */
static int send_ci_msg(msg_queue* q, vci_msg msg) {
    assert(msg.type == VCI_OK || msg.type == VCI_STATUS || msg.type == VCI_IS_PRESENT || msg.type == VCI_ERR || msg.type == VCI_SYS_ERR);

    if (sendto(q->sock, &msg, sizeof(msg), 0, (void*) &q->resp_addr, q->resp_len) < 0) {
        return -1;
    }

    log_ci_msg("SEND", q->resp_addr.sun_path, msg);
    return 0;
}

/* close socket associated with the message queue */
static void close_msg_queue(msg_queue* q) {
    close(q->sock);
}

static void switch_mux_for_rank(struct dpu_rank_t* rank, bool switch_for_host) {
    dpu_error_t dpu_switch_mux_for_rank(struct dpu_rank_t* rank, bool set_mux_for_host);

    DPU_ASSERT(dpu_switch_mux_for_rank(rank, switch_for_host));
}

static void status_for_ci(struct dpu_rank_t* rank, uint8_t ci_nr, uint8_t* out_done, uint8_t* out_fault) {
    DPU_ASSERT(dpu_poll_rank(rank));

    dpu_run_context_t ctx = dpu_get_run_context(rank);

    *out_done = ~ctx->dpu_running[ci_nr];
    *out_fault = ctx->dpu_in_fault[ci_nr];
}

static dpu_error_t custom_debug_teardown_dpu(struct dpu_t* dpu, struct dpu_context_t* context, dpu_thread_t fault_thread) {
    uint32_t ufi_select_dpu(struct dpu_rank_t* rank, uint8_t* mask, uint8_t dpu);
    uint32_t ufi_set_dpu_fault_and_step(struct dpu_rank_t *rank, uint8_t ci_mask);
    uint32_t ufi_set_bkp_fault(struct dpu_rank_t *rank, uint8_t ci_mask);
    dpu_error_t ci_start_thread_dpu(struct dpu_t *dpu, dpu_thread_t thread, bool resume, uint8_t *previous);
    uint32_t ufi_read_bkp_fault(struct dpu_rank_t *rank, uint8_t ci_mask, uint8_t *fault);
    uint32_t ufi_clear_fault_bkp(struct dpu_rank_t *rank, uint8_t ci_mask);
    uint32_t ufi_clear_fault_dpu(struct dpu_rank_t *rank, uint8_t ci_mask);

    struct dpu_rank_t* rank = dpu_get_rank(dpu);
    dpu_slice_id_t slice_id = dpu_get_slice_id(dpu);
    dpu_member_id_t member_id = dpu_get_member_id(dpu);

    dpu_error_t status;
    uint8_t nr_of_threads_per_dpu = 24;
    uint8_t nr_of_running_threads = context->nr_of_running_threads;
    dpu_thread_t scheduling_order[nr_of_threads_per_dpu];
    dpu_thread_t each_thread;
    dpu_thread_t each_running_thread;

    uint8_t mask = 1 << slice_id;

    if ((status = ufi_select_dpu(rank, &mask, member_id)) != DPU_OK) {
        return status;
    }

    // 1. Set fault & bkp_fault
    if ((status = ufi_set_dpu_fault_and_step(rank, mask)) != DPU_OK) {
        return status;
    }

    if ((status = ufi_set_bkp_fault(rank, mask)) != DPU_OK) {
        return status;
    }

    // 2. Resume running threads
    for (each_thread = 0; each_thread < nr_of_threads_per_dpu;
         ++each_thread) {
        uint8_t scheduling_position = context->scheduling[each_thread];
        if (scheduling_position != 0xFF) {
            scheduling_order[scheduling_position] = each_thread;
        }
    }

    for (each_running_thread = 0;
         each_running_thread < nr_of_running_threads;
         ++each_running_thread) {

        dpu_thread_t tid = scheduling_order[each_running_thread];
        bool reset_pc = tid == fault_thread;

        if ((status = ci_start_thread_dpu(dpu, tid, !reset_pc, NULL)) != DPU_OK) {
            return status;
        }
    }
    // Interception Fault Clear
    if ((status = ufi_read_bkp_fault(rank, mask, NULL)) != DPU_OK) {
        return status;
    }

    // 3. Clear bkp_fault & fault
    if ((status = ufi_clear_fault_bkp(rank, mask)) != DPU_OK) {
        return status;
    }

    /* If the DPU was in fault (according to the context), we need to keep it in fault
     * so that any other process (mostly a host application) will see it in this state.
     */
    if (!((context->bkp_fault && context->bkp_fault_id != 0) ||
          context->mem_fault || context->dma_fault)) {
        if ((status = ufi_clear_fault_dpu(rank, mask)) != DPU_OK) {
            return status;
        }
    }

    return DPU_OK;
}

static dpu_error_t custom_finalize_fault_process_for_dpu(struct dpu_t* dpu, struct dpu_context_t* ctx, dpu_thread_t fault_thread) {
    struct dpu_rank_t* rank = dpu_get_rank(dpu);
    dpu_error_t status = DPU_OK;

    dpu_lock_rank(rank);

    status = custom_debug_teardown_dpu(dpu, ctx, fault_thread);

    dpu_unlock_rank(rank);
    return status;
}

static unsigned short get_rank_id(const struct dpu_rank_t* rank) {
    const short* ptr = (short*)((uint8_t*) rank + 4);
    unsigned res = (*ptr) & 0xFFF;

    assert(res < 40);
    return res;
}

static void reset_for_rank(struct dpu_rank_t* rank, bool set_crypt_regs) {
    struct dpu_context_t* ctx = calloc(MAX_NR_DPUS_PER_RANK, sizeof(*ctx));
    assert(ctx != NULL);

    for (int i = 0; i < MAX_NR_DPUS_PER_RANK; ++i) {
        struct dpu_t* dpu = dpu_get(rank, i / 8, i % 8);
        DPU_ASSERT(dpu_context_fill_from_rank(&ctx[i], rank));

        DPU_ASSERT(dpu_initialize_fault_process_for_dpu(dpu, &ctx[i], 0x1000));

        if (set_crypt_regs) {
            DPU_ASSERT(dpu_extract_context_for_dpu(dpu, &ctx[i]));

            for (int j = 0; j < 24; ++j) {
                for (int k = 0; k < 4; ++k) {
                    ctx[i].registers[14 + k + 24 * j] = 0;
                    ctx[i].registers[18 + k + 24 * j] = 0x10101010;
                }
            }

            DPU_ASSERT(dpu_restore_context_for_dpu(dpu, &ctx[i]));
        }

        dpu_thread_t tid = ctx[i].bkp_fault_thread_index;
        unsigned res = ctx[i].bkp_fault_id;

        /* TODO: Allow more flexibility in fault codes */
        assert(ctx[i].bkp_fault && res == 0x101010);

        /* TODO: Proper concept for multithreading */
        for (int j = 0; j < 24; ++j) {
            if (j != tid) {
                ctx[i].scheduling[j] = 0xFF;
            }
        }

        ctx[i].bkp_fault = false;
        ctx[i].bkp_fault_id = 0;
    }

    for (int i = 0; i < MAX_NR_DPUS_PER_RANK; ++i) {
        dpu_thread_t tid = ctx[i].bkp_fault_thread_index;
        struct dpu_t* dpu = dpu_get(rank, i / 8, i % 8);

        DPU_ASSERT(custom_finalize_fault_process_for_dpu(dpu, &ctx[i], tid));
        dpu_free_dpu_context(&ctx[i]);
    }

    DPU_ASSERT(dpu_poll_rank(rank));
    free(ctx);
}

static int parse_cli_args(int argc, char** argv, cli_args* out_args) {
    *out_args = (cli_args) {
        .nr_ranks = 1
    };

    for (int i = 1; i < argc; ++i) {
        const char* cur = argv[i];
        size_t len = strlen(cur);

        if (len < 2) {
            printf("[FAIL] Invalid argument syntax: \"%s\"", cur);
            return -1;
        }

        cur += 2;
        len -= 2;

        const char* val = strchr(cur, '=');

        if (val == NULL) {
            printf("[FAIL] Invalid argument syntax: \"%s\"\n", cur);
            return -1;
        }

        val += 1;

        const char* key = cur;
        size_t val_len = strlen(val);
        size_t key_len = len - val_len - 1;

        if (key_len == sizeof("nr_ranks") - 1 && memcmp(key, "nr-ranks", key_len) == 0) {
            char* end_ptr = NULL;
            long nr_ranks = strtol(val, &end_ptr, 10);

            if (end_ptr == val || *end_ptr != '\0' || !isdigit(val[0])) {
                printf("[FAIL] Invalid value for option --nr-ranks\n");
                return -1;
            }

            if (nr_ranks <= 0 || nr_ranks >= 40) {
                printf("[FAIL] Value out of range for option --nr-ranks\n");
                return -1;
            }

            out_args->nr_ranks = nr_ranks;
        } else {
            printf("[FAIL] Unkown option --%.*s\n", (int) key_len, key);
            return -1;
        }
    }

    return 0;
}

static void print_hexdump(size_t n, const uint8_t* buf) {
    FILE* pp = popen("xxd", "w");

    if (pp == NULL) {
        perror("popen()");
        return;
    }

    if (fwrite(buf, 1, n, pp) < n) {
        perror("fwrite()");
    }

    pclose(pp);
}

static void wait_for_faults(struct dpu_rank_t* rank) {
    uint64_t done, fault;

    do {
        done = 0;
        fault = 0;

        uint8_t done_u8;
        uint8_t fault_u8;

        for (int i = 0; i < 8; ++i) {
            status_for_ci(rank, i, &done_u8, &fault_u8);

            done |= done_u8 << (i * 8);
            fault |= fault_u8 << (i * 8);
        }

        usleep(1000);
    } while ((done | fault ) != UINT64_MAX);
}

static void init_crypto_regs(switch_state* state) {
    for (int i = 0; i < 40; ++i) {
        if (state->ranks[i].rank == NULL) {
            continue;
        }

        wait_for_faults(state->ranks[i].rank);
    }

    for (int i = 0; i < 40; ++i) {
        if (state->ranks[i].rank == NULL) {
            continue;
        }

        reset_for_rank(state->ranks[i].rank, true);
    }

    for (int i = 0; i < 40; ++i) {
        if (state->ranks[i].rank == NULL) {
            continue;
        }

        wait_for_faults(state->ranks[i].rank);
    }
}

static int deploy_dpu_ids(struct dpu_rank_t* r, struct dpu_program_t* prog) {
    struct dpu_symbol_t sym;
    int err;

    err = dpu_get_symbol(prog, "vault_get_id", &sym);

    if (err) {
        return err;
    }

    unsigned rank_id = get_rank_id(r);

    for (int i = 0; i < 64; ++i) {
        unsigned id = rank_id * 64 + i;
        struct dpu_t* dpu = dpu_get(r, i / 8, i % 8);

        uint64_t buf[2] = {
            0x0000606300000000,
            0x00008c5f00000000
        };

        buf[0] |= (id & 0xF) << 20;
        buf[0] |= (id >> 4 & 0xF) << 16;

        uint8_t msn = (id >> 8 & 0xF);

        msn = (msn & 0x8) >> 3 | (msn & 0x4) >> 1 | (msn & 0x2) << 1 | (msn & 0x1) << 3;
        buf[0] |= msn << 12;

        err = dpu_copy_to_iram_for_dpu(dpu, (sym.address - 0x80000000) / 8, buf, 2);

        if (err) {
            return err;
        }
    }

    return DPU_OK;
}

int main(int argc, char** argv) {
    cli_args args;

    if (parse_cli_args(argc, argv, &args) < 0) {
        return EXIT_FAILURE;
    }

    msg_queue q = init_unix_socket();

    if (q.sock < 0) {
        return EXIT_FAILURE;
    }

    struct dpu_set_t set, rank;
    struct dpu_program_t* prog;
    switch_state state = { 0 };

    DPU_ASSERT(dpu_alloc_ranks(args.nr_ranks, s_dpu_profile, &set));
    DPU_ASSERT(dpu_load(set, "./fault", &prog));

    DPU_RANK_FOREACH(set, rank) {
        struct dpu_rank_t* r = rank.list.ranks[0];
        DPU_ASSERT(deploy_dpu_ids(r, prog));
    }

    DPU_ASSERT(dpu_launch(set, DPU_ASYNCHRONOUS));

    DPU_RANK_FOREACH(set, rank) {
        struct dpu_rank_t* r = rank.list.ranks[0];
        unsigned short id = get_rank_id(r);

        printf("[INFO] Allocated rank %u\n", id);
        state.ranks[id].rank = r;
    }

    init_crypto_regs(&state);
    puts("All ranks up and running.");

    while (!s_sig_term_received) {
        vci_msg msg = recv_ci_msg(&q);

        if (msg.type < 0) {
            log_ci_msg("FAIL", NULL, msg);
            continue;
        }

        // mirror rank number in responses to allow multiple ranks to use a single socket
        vci_msg resp = {
                .type = VCI_OK,
                .rank_nr = msg.rank_nr
        };

        if (msg.rank_nr >= 40 || state.ranks[msg.rank_nr].rank == NULL) {
            printf("[FAIL] received vci message for invalid rank number\n");

            resp.type = VCI_ERR;

            if (send_ci_msg(&q, resp) < 0) {
                printf("[FAIL] Cannot send message: %s\n", strerror(errno));
            }

            continue;
        }

        switch (msg.type) {
        case VCI_PRESENT:
            // not present case already handled above
            resp.type = VCI_IS_PRESENT;
            break;

        case VCI_ACQ_MUX:
            // The poll_rank call is required, because otherwise switch_mux_for_rank won't know that the DPUs
            // are in fault and prevent switching the MUX.
            DPU_ASSERT(dpu_poll_rank(state.ranks[msg.rank_nr].rank));
            switch_mux_for_rank(state.ranks[msg.rank_nr].rank, true);
            break;

        case VCI_REL_MUX:
            switch_mux_for_rank(state.ranks[msg.rank_nr].rank, false);
            break;

        case VCI_GET_STATUS:
            status_for_ci(state.ranks[msg.rank_nr].rank, msg.ci_nr, &resp.done_bits, &resp.fault_bits);

            resp.type = VCI_STATUS;
            break;

        case VCI_RST_DPUS:
            reset_for_rank(state.ranks[msg.rank_nr].rank, false);
            break;

        // TODO: Undo temporary extension
        case VCI_RST_DPUS + 1:
            switch_mux_for_rank(state.ranks[msg.rank_nr].rank, true);

            struct dpu_t* dpu = dpu_get(state.ranks[msg.rank_nr].rank, 0, 0);
            uint64_t buf[2] = { 0 };

            DPU_ASSERT(dpu_copy_from_mram(dpu, (uint8_t*) &buf[0], 0, sizeof(buf)));
            print_hexdump(sizeof(buf), (const uint8_t*) &buf[0]);

            break;
        }

        if (send_ci_msg(&q, resp) < 0) {
            printf("[FAIL] Cannot send message: %s\n", strerror(errno));
        }
    }

    dpu_free(set);
    close_msg_queue(&q);

    return 0;
}
