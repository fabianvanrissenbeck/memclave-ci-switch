#include "vci-msg.h"

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
#include <dpu_management.h>
#include <dpu_transfer_matrix.h>

typedef struct msg_queue {
    int sock;
    struct sockaddr_un resp_addr;
    socklen_t resp_len;
} msg_queue;

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

    if (res.type < 0 || res.type == VCI_STATUS || res.type == VCI_OK || res.type > VCI_RST_DPUS || res.ci_nr >= 8) {
        res.type = VCI_MSG_ERR;
    }

    return res;
}

/** validate and sent a message to the socket */
static int send_ci_msg(msg_queue* q, vci_msg msg) {
    assert(msg.type == VCI_OK || msg.type == VCI_STATUS || msg.type == VCI_IS_PRESENT || msg.type == VCI_ERR || msg.type == VCI_SYS_ERR);

    if (sendto(q->sock, &msg, sizeof(msg), 0, (void*) &q->resp_addr, q->resp_len) < 0) {
        return -1;
    }

    return 0;
}

/* close socket associated with the message queue */
static void close_msg_queue(msg_queue* q) {
    close(q->sock);
}

/** print out message in readable format */
static void log_ci_msg(vci_msg msg) {
    printf(
        "{ .type = %d, .ci_nr = %u, .done_bits = %u, .fault_bits = %u, .rank_nr = %u }\n",
        msg.type, msg.ci_nr, msg.done_bits, msg.fault_bits, msg.rank_nr
    );
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

static void reset_for_rank(struct dpu_rank_t* rank) {
    struct dpu_context_t* ctx = calloc(MAX_NR_DPUS_PER_RANK, sizeof(*ctx));
    assert(ctx != NULL);

    for (int i = 0; i < MAX_NR_DPUS_PER_RANK; ++i) {
        struct dpu_t* dpu = dpu_get(rank, i / 8, i % 8);
        DPU_ASSERT(dpu_context_fill_from_rank(&ctx[i], rank));

        /* TODO: Figure out what that last argument does */
        DPU_ASSERT(dpu_initialize_fault_process_for_dpu(dpu, &ctx[i], 0x1000));

        DPU_ASSERT(dpu_extract_pcs_for_dpu(dpu, &ctx[i]));
        DPU_ASSERT(dpu_extract_context_for_dpu(dpu, &ctx[i]));

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

int main() {
    msg_queue q = init_unix_socket();

    if (q.sock < 0) {
        return EXIT_FAILURE;
    }

    struct dpu_set_t set;

    DPU_ASSERT(dpu_alloc_ranks(1, s_dpu_profile, &set));
    DPU_ASSERT(dpu_load(set, "./fault", NULL));
    DPU_ASSERT(dpu_launch(set, DPU_ASYNCHRONOUS));

    while (!s_sig_term_received) {
        vci_msg msg = recv_ci_msg(&q);

        if (msg.type < 0) {
            printf("[FAIL] received invalid message: ");
            log_ci_msg(msg);

            continue;
        }

        printf("[RECV] ");
        log_ci_msg(msg);

        vci_msg resp = { VCI_OK };

        switch (msg.type) {
        case VCI_PRESENT:
            if (msg.rank_nr == 0) {
                resp.type = VCI_IS_PRESENT;
            } else {
                resp.type = VCI_ERR;
            }

            break;

        case VCI_ACQ_MUX:
            // right now only one rank is supported
            switch_mux_for_rank(set.list.ranks[0], true);
            break;

        case VCI_REL_MUX:
            switch_mux_for_rank(set.list.ranks[0], false);
            break;

        case VCI_GET_STATUS:
            status_for_ci(set.list.ranks[0], msg.ci_nr, &resp.done_bits, &resp.fault_bits);

            resp.type = VCI_STATUS;
            break;

        case VCI_RST_DPUS:
            reset_for_rank(set.list.ranks[0]);
            break;
        }

        printf("[SEND] ");
        log_ci_msg(resp);

        if (send_ci_msg(&q, resp) < 0) {
            printf("[FAIL] Cannot send message: %s\n", strerror(errno));
        }
    }

    dpu_free(set);
    close_msg_queue(&q);

    return 0;
}
