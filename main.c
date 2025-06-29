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

    if (res.type < 0 || res.type == VCI_QRY_RES || res.type == VCI_OK || res.type > VCI_REL_MUX || res.ci_nr >= 8) {
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

static void switch_mux_for_line(struct dpu_rank_t* rank, uint8_t line_id, uint8_t mask) {
    dpu_error_t dpu_switch_mux_for_dpu_line(struct dpu_rank_t*, uint8_t, uint8_t);
    DPU_ASSERT(dpu_switch_mux_for_dpu_line(rank, line_id, mask));
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

static dpu_error_t clear_and_continue(struct dpu_t* dpu, iram_addr_t* out_addr) {
    struct dpu_context_t* ctx = calloc(MAX_NR_DPUS_PER_RANK, sizeof(*ctx));
    assert(ctx != NULL);

    dpu_error_t err = DPU_OK;

    if ((err = dpu_context_fill_from_rank(&ctx[0], dpu_get_rank(dpu)))) {
        goto fail;
    }

    if ((err = dpu_initialize_fault_process_for_dpu(dpu, &ctx[0], 0x1000))) {
        goto fail;
    }

    if ((err = dpu_extract_pcs_for_dpu(dpu, &ctx[0]))) {
        goto fail;
    }

    dpu_thread_t tid = ctx[0].bkp_fault_thread_index;
    iram_addr_t pc = ctx[0].pcs[tid];

    if (out_addr) {
        *out_addr = pc;
    }

    if ((err = dpu_finalize_fault_process_for_dpu(dpu, &ctx[0]))) {
        goto fail;
    }

    dpu_free_dpu_context(&ctx[0]);

    fail:
        free(ctx);
    return err;
}

static dpu_error_t wait_for_fault(struct dpu_t* dpu) {
    bool is_running, fault;
    dpu_error_t err = DPU_OK;
    uint64_t timeout = 0;

    do {
        if ((err = dpu_poll_dpu(dpu, &is_running, &fault))) {
            return err;
        }

        usleep(timeout);
        timeout = timeout ? timeout << 1 : 1;
    } while (!fault && is_running);

    if (!is_running && !fault) {
        puts("DPU finished while waiting for fault.");
        return DPU_ERR_DPU_FAULT;
    }

    return DPU_OK;
}

static dpu_error_t dpu_simple_step_over(struct dpu_t* dpu) {
    dpuinstruction_t buf[2];

    dpuinstruction_t repl_buf[2] = {
        0x7c6300000000,
        0x7e6320110101
    };

    iram_addr_t pc;

    DPU_ASSERT(clear_and_continue(dpu, &pc));
    DPU_ASSERT(wait_for_fault(dpu));
    DPU_ASSERT(dpu_copy_from_iram_for_dpu(dpu, buf, pc, 2));
    DPU_ASSERT(dpu_copy_to_iram_for_dpu(dpu, pc, repl_buf, 2));
    DPU_ASSERT(clear_and_continue(dpu, NULL));
    DPU_ASSERT(wait_for_fault(dpu));
    DPU_ASSERT(dpu_copy_to_iram_for_dpu(dpu, pc, buf, 2));
    DPU_ASSERT(clear_and_continue(dpu, NULL));

    return DPU_OK;
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

static uint8_t mux_of_line(switch_state* state, uint8_t rank_nr, uint8_t line) {
    assert(rank_nr < 40);
    assert(state->ranks[rank_nr].rank != NULL);

    uint8_t mask = 0;

    for (uint8_t ci = 0; ci < 8; ci++) {
        uint8_t idx = tuple_to_index(ci, line);
        mask |= ((state->ranks[rank_nr].mux_state & (1llu << idx)) != 0) << ci;
    }

    return mask;
}

static void switch_state_update_mux(switch_state* state, uint8_t rank_nr) {
    assert(rank_nr < 40);
    assert(state->ranks[rank_nr].rank != NULL);

    for (uint8_t line = 0; line < 8; line++) {
        uint8_t mask = mux_of_line(state, rank_nr, line);
        switch_mux_for_line(state->ranks[rank_nr].rank, line, mask);
    }
}

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

static void continue_execution_of(switch_state* state, uint8_t rank_nr, uint8_t line_nr) {
    assert(rank_nr < 40);
    assert(state->ranks[rank_nr].rank != NULL);

    for (int i = 0; i < 8; i++) {
        uint8_t idx = tuple_to_index(idx, line_nr);
        state->ranks[rank_nr].mux_state &= ~((uint64_t)(1) << idx);
    }

    switch_state_update_mux(state, rank_nr);

    for (int i = 0; i < 8; ++i) {
        struct dpu_t* dpu = dpu_get(state->ranks[rank_nr].rank, i, line_nr);

        bool is_done, is_faulted;
        DPU_ASSERT(dpu_status_dpu(dpu, &is_done, &is_faulted));

        assert(is_faulted);
        DPU_ASSERT(dpu_simple_step_over(dpu));
    }
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
    DPU_ASSERT(dpu_load(set, "../fault", &prog));
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
            continue_execution_of(&state, msg.rank_nr, msg.line_nr);

            break;

        case VCI_QRY_MUX:
            switch_state_update_rank(&state, msg.rank_nr);

            resp.type = VCI_QRY_RES;
            resp.resp = mux_of_line(&state, msg.rank_nr, msg.line_nr);

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
