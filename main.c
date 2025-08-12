#include "common/vci-msg.h"

#include <time.h>
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
        /** bit 1 is set if the MUX of DPU i is facing the host */
        uint64_t mux_state;
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

    if (res.type < 0 || res.type == VCI_QRY_RES || res.type == VCI_OK || res.type > VCI_REL_MUX) {
        res.type = VCI_MSG_ERR;
    }

    log_ci_msg("RECV", q->resp_addr.sun_path, res);
    return res;
}

/** validate and sent a message to the socket */
static int send_ci_msg(msg_queue* q, vci_msg msg) {
    assert(msg.type == VCI_OK || msg.type == VCI_QRY_RES || msg.type == VCI_IS_PRESENT || msg.type == VCI_ERR || msg.type == VCI_SYS_ERR);

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

static void switch_mux_for_rank(struct dpu_rank_t* rank, bool set_mux_for_host) {
    dpu_error_t dpu_switch_mux_for_rank(struct dpu_rank_t *, bool);
    DPU_ASSERT(dpu_switch_mux_for_rank(rank, set_mux_for_host));
}

static unsigned short get_rank_id(const struct dpu_rank_t* rank) {
    const short* ptr = (short*)((uint8_t*) rank + 4);
    unsigned res = (*ptr) & 0xFFF;

    assert(res < 40);
    return res;
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

/**
 * @brief convert from CI and line number to an index between 0 and 63
 * @param slice_id CI number of the DPU
 * @param member_id line number of the DPU
 * @return index between 0 and 63
 */
static uint8_t tuple_to_index(uint8_t slice_id, uint8_t member_id) {
    return 8 * slice_id + member_id;
}

static void switch_state_update_mux(switch_state* state, uint8_t rank_nr) {
    assert(rank_nr < 40);
    assert(state->ranks[rank_nr].rank != NULL);

    if (state->ranks[rank_nr].mux_state == UINT64_MAX) {
        switch_mux_for_rank(state->ranks[rank_nr].rank, true);
    }
}

/*
 *
 * MUX weirdness
 *
 * There are DPU pairs. These are DPUs on different lines with the same CI to the best
 * of my understanding. Both DPUs are switched simultaniously.
 *
 * General Concept
 *
 * + Maintain one host and one guest view of the DPU status. The host view specifies which DPUs are
 *   currently in fault, the guest view specifies which can actually be written two. These
 *   can be different, because one DPU of a pair may be in fault, but its MUX cannot be switched
 *   because the pair is still running.
 * + If two DPUs of a pair are in fault but not yet switched, switch them and add this info to the guest view.
 * + When resolving a fault, the ci-switch must remove the pair DPU from the
 *
 */

static void switch_state_update_rank(switch_state* state, uint8_t rank_nr) {
    assert(rank_nr < 40);
    assert(state->ranks[rank_nr].rank != NULL);

    struct dpu_rank_t* rank = state->ranks[rank_nr].rank;
    DPU_ASSERT(dpu_poll_rank(rank));

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            struct dpu_t* dpu = dpu_get(rank, i, j);

            bool is_done, is_faulted;
            DPU_ASSERT(dpu_status_dpu(dpu, &is_done, &is_faulted));

            uint8_t idx = tuple_to_index(i, j);

            state->ranks[rank_nr].mux_state &= ~((uint64_t)(1) << idx);
            state->ranks[rank_nr].mux_state |= (uint64_t)(is_faulted) << idx;
        }
    }

    switch_state_update_mux(state, rank_nr);
}

static uint64_t get_time_us(void) {
    struct timespec ts;

    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
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

        // printf("[DPU %p] Resuming thread %d\n", dpu, tid);
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

static void reset_for_dpu(struct dpu_t* dpu) {
    struct dpu_context_t* ctx = calloc(MAX_NR_DPUS_PER_RANK, sizeof(*ctx));
    assert(ctx != NULL);

    DPU_ASSERT(dpu_context_fill_from_rank(&ctx[0], dpu_get_rank(dpu)));

    for (int i = 0; i < 24; ++i) {
        ctx[0].scheduling[i] = 0xFF;
    }

#if 0
    uint64_t start_time = get_time_us();
    DPU_ASSERT(dpu_initialize_fault_process_for_dpu(dpu, &ctx[0], 0x1000));

    int nr_thread_running = 0;
    int thread_running = -1;

    for (int i = 0; i < 24; ++i) {
        if (ctx[0].scheduling[i] != 0xFF) {
            nr_thread_running++;
            thread_running = i;
        }
    }

    printf("[TIME] Time elapsed: %.2f Threads Running: %d Thread Running: %d\n", (double)(get_time_us() - start_time) / 1000.0, nr_thread_running, thread_running);

    unsigned res = ctx[0].bkp_fault_id;

    if (res != 0x101010) {
        printf("Received unexpected fault with code %02x\n", res);
    }

    assert(ctx[0].bkp_fault && res == 0x101010);

    ctx[0].bkp_fault = false;
    ctx[0].bkp_fault_id = 0;

#if 0
    // if (dpu == dpu_get(dpu_get_rank(dpu), 0, 0)) {
        dpu_extract_context_for_dpu(dpu, &ctx[0]);
    // }
#endif

    dpu_thread_t tid = ctx[0].bkp_fault_thread_index;
#endif

    dpu_thread_t tid = 0;
    ctx[0].scheduling[0] = 0;

    DPU_ASSERT(custom_finalize_fault_process_for_dpu(dpu, &ctx[0], tid));

    dpu_free_dpu_context(&ctx[0]);
    free(ctx);
}

static void continue_execution_of(switch_state* state, uint8_t rank_nr) {
    assert(rank_nr < 40);
    assert(state->ranks[rank_nr].rank != NULL);

    switch_mux_for_rank(state->ranks[rank_nr].rank, false);
    state->ranks[rank_nr].mux_state = 0;

    for (int i = 0; i < 64; ++i) {
        struct dpu_t* dpu = dpu_get(state->ranks[rank_nr].rank, i / 8, i % 8);

        bool is_done, is_faulted;
        DPU_ASSERT(dpu_status_dpu(dpu, &is_done, &is_faulted));

        assert(is_faulted);
        reset_for_dpu(dpu);
    }
}

static uint8_t count_bits_of(uint64_t n) {
    uint8_t res = 0;

    while (n) {
        res += n & 1;
        n >>= 1;
    }

    return res;
}

static uint8_t* load_file_from(const char* path, size_t* out_size) {
    FILE* fp = fopen(path, "rb");
    assert(fp != NULL);

    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    uint8_t* res = malloc(size);
    assert(res != NULL);

    fread(res, size, 1, fp);
    fclose(fp);

    *out_size = size;
    return res;
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

    struct dpu_set_t set, rank, dpu;
    struct dpu_program_t* prog;
    switch_state state = { 0 };

    DPU_ASSERT(dpu_alloc_ranks(args.nr_ranks, s_dpu_profile, &set));
    DPU_ASSERT(dpu_load(set, "./reset.elf", NULL));
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    DPU_ASSERT(dpu_load(set, "../ime", &prog));

    size_t msg_sz;
    uint8_t* msg = load_file_from("../msg.sk", &msg_sz);

    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_to_mram(dpu.dpu, 63 << 20, msg, msg_sz));
    }

    free(msg);

    DPU_ASSERT(dpu_launch(set, DPU_ASYNCHRONOUS));

    DPU_RANK_FOREACH(set, rank) {
        struct dpu_rank_t* r = rank.list.ranks[0];
        unsigned short id = get_rank_id(r);

        printf("[INFO] Allocated rank %u\n", id);
        state.ranks[id].rank = r;
    }

    puts("All ranks up and running.");

    while (!s_sig_term_received) {
        vci_msg msg = recv_ci_msg(&q);

        if (msg.type < 0) {
            log_ci_msg("FAIL", NULL, msg);
            continue;
        }

        uint64_t start_time = get_time_us();

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

        case VCI_REL_MUX:
            switch_state_update_rank(&state, msg.rank_nr);
            continue_execution_of(&state, msg.rank_nr);

            break;

        case VCI_QRY_MUX:
            switch_state_update_rank(&state, msg.rank_nr);

            resp.type = VCI_QRY_RES;
            resp.n_faulted = count_bits_of(state.ranks[msg.rank_nr].mux_state);
            resp.n_running = 64 - resp.n_faulted;

            break;
        }

        uint64_t elapsed = get_time_us() - start_time;

        if (send_ci_msg(&q, resp) < 0) {
            printf("[FAIL] Cannot send message: %s\n", strerror(errno));
        }

        printf("[TIME] Time elapsed: %.2fms\n", (double) elapsed / 1000.0);
    }

    dpu_free(set);
    close_msg_queue(&q);

    return 0;
}
