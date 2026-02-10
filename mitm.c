#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <getopt.h>
#include <err.h>
#include <assert.h>
#include <string.h>

#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif


/***************************** definition of structs *************************/

#define BUFFER_SIZE 10000       
#define CHECK_INTERVAL 10       
#define RECV_SLOTS 8            

#define TAG_FILL 100            // Tag phase 1
#define TAG_PROBE_WORK 200      // Tag phase 2

typedef uint64_t u64;           // portable 64-bit integer
typedef uint32_t u32;           // portable 32-bit integer 

typedef uint32_t u32i;          // portable 32-bit integer for metadata

struct __attribute__ ((packed)) entry { u32 k; u64 v; };  // hash table entry 

struct __attribute__ ((packed)) msg_entry { 
    u64 z;                      // fx in phase 1 or gy in phase 2
    u64 x;                      // x in phase 1 or y in phase 2
};

struct msg_buffer {
    struct msg_entry data[BUFFER_SIZE];
    int count;
};

struct send_slot {
    struct msg_buffer buf;
    MPI_Request req;
    int active;   // 0 = free, 1 = Isend active
};

struct __attribute__ ((packed)) solution_rec {
    u64 k1;
    u64 k2;
    u32i rank;
    u32i partition;
};

struct part_timing {
    double fill_min, fill_avg, fill_max;
    double probe_min, probe_avg, probe_max;
};

/***************************** global variables ******************************/

u64 n = 0;                      // block size (in bits) 
u64 mask;                       // this is 2**n - 1 

u64 dict_size;                  // number of slots in the hash table 
u32 *A_k;                       // hash table keys
u64 *A_v;                       // hash table values
 
static int verbose = 0;

// CLI-controlled slicing. Default: disabled (1 partition).
static int num_partitions = 1;

/* (P, C) : two plaintext-ciphertext pairs */
u32 P[2][2] = {{0, 0}, {0xffffffff, 0xffffffff}};
u32 C[2][2];


/************************ tools and utility functions *************************/

double wtime()
{
	struct timeval ts;
	gettimeofday(&ts, NULL);
	return (double)ts.tv_sec + ts.tv_usec / 1E6;
}

static int cmp_solution_rec(const void *a, const void *b)
{
    const struct solution_rec *ra = (const struct solution_rec *)a;
    const struct solution_rec *rb = (const struct solution_rec *)b;

    if (ra->partition < rb->partition) return -1;
    if (ra->partition > rb->partition) return 1;

    if (ra->rank < rb->rank) return -1;
    if (ra->rank > rb->rank) return 1;

    if (ra->k1 < rb->k1) return -1;
    if (ra->k1 > rb->k1) return 1;
    if (ra->k2 < rb->k2) return -1;
    if (ra->k2 > rb->k2) return 1;
    return 0;
}

static void final_recap_print(int my_rank,
                              int world_size,
                              int partitions,
                              int maxres,
                              const struct solution_rec *local,
                              int local_n,
                              u64 overflow_mask,
                              int overflow_any,
                              const struct part_timing *timings,
                              double total_effective_time)
{
    // Gather counts.
    int *counts = NULL;
    if (my_rank == 0)
        counts = (int *)malloc((size_t)world_size * sizeof(int));

    MPI_Gather(&local_n, 1, MPI_INT,
               counts, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    // Gather overflow indicators.
    u64 *masks = NULL;
    int *anys = NULL;
    if (my_rank == 0) {
        masks = (u64 *)malloc((size_t)world_size * sizeof(u64));
        anys = (int *)malloc((size_t)world_size * sizeof(int));
    }
    MPI_Gather(&overflow_mask, 1, MPI_UINT64_T,
               masks, 1, MPI_UINT64_T,
               0, MPI_COMM_WORLD);
    MPI_Gather(&overflow_any, 1, MPI_INT,
               anys, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    // Gather solution records as bytes.
    int total = 0;
    int *displs = NULL;
    struct solution_rec *all = NULL;

    if (my_rank == 0) {
        displs = (int *)malloc((size_t)world_size * sizeof(int));
        int off = 0;
        for (int r = 0; r < world_size; r++) {
            displs[r] = off;
            off += counts[r];
        }
        total = off;
        if (total > 0)
            all = (struct solution_rec *)malloc((size_t)total * sizeof(struct solution_rec));
    }

    // Convert counts/displs to bytes for MPI_Gatherv.
    int send_bytes = local_n * (int)sizeof(struct solution_rec);
    int *counts_b = NULL;
    int *displs_b = NULL;
    if (my_rank == 0) {
        counts_b = (int *)malloc((size_t)world_size * sizeof(int));
        displs_b = (int *)malloc((size_t)world_size * sizeof(int));
        for (int r = 0; r < world_size; r++) {
            counts_b[r] = counts[r] * (int)sizeof(struct solution_rec);
            displs_b[r] = displs[r] * (int)sizeof(struct solution_rec);
        }
    }

    MPI_Gatherv((const void *)local, send_bytes, MPI_BYTE,
                (void *)all, counts_b, displs_b, MPI_BYTE,
                0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        // Stable recap: sort by partition, then rank.
        if (total > 1)
            qsort(all, (size_t)total, sizeof(struct solution_rec), cmp_solution_rec);

        // De-duplicate identical solution records (can happen due to repeated discovery/processing).
        // Uniqueness key: (partition, rank, k1, k2) as defined by cmp_solution_rec.
        if (total > 1) {
            int uniq = 1;
            for (int i = 1; i < total; i++) {
                if (cmp_solution_rec(&all[i], &all[uniq - 1]) != 0) {
                    all[uniq++] = all[i];
                }
            }
            total = uniq;
        }

         printf("\n==================== FINAL RECAP ====================\n");
         printf("Run summary: MPI ranks=%d, partitions=%d, maxres(per rank per partition)=%d\n",
             world_size, partitions, maxres);

         if (timings != NULL) {
             printf("\nTiming recap (min/avg/max across ranks; effective ~= max(fill)+max(probe)):\n");
             for (int pidx = 0; pidx < partitions; pidx++) {
              const double eff = timings[pidx].fill_max + timings[pidx].probe_max;
              if (partitions > 1) {
                  printf("  - partition %d/%d:\n", pidx + 1, partitions);
              } else {
                  printf("  - single pass:\n");
              }
              printf("      fill : %.3fs / %.3fs / %.3fs\n",
                  timings[pidx].fill_min, timings[pidx].fill_avg, timings[pidx].fill_max);
              printf("      probe: %.3fs / %.3fs / %.3fs\n",
                  timings[pidx].probe_min, timings[pidx].probe_avg, timings[pidx].probe_max);
              printf("      effective (partition): %.3fs\n", eff);
             }
             printf("Total effective time (sum of per-partition effective times): %.3fs\n",
                 total_effective_time);
         }
        printf("Solutions recorded (may be truncated on overflow): %d\n", total);

        int any_overflow = 0;
        if (partitions <= 64) {
            for (int r = 0; r < world_size; r++) {
                if (masks && masks[r]) {
                    any_overflow = 1;
                    break;
                }
            }
        } else {
            for (int r = 0; r < world_size; r++) {
                if (anys && anys[r]) {
                    any_overflow = 1;
                    break;
                }
            }
        }

        if (any_overflow) {
            printf("WARNING: overflow occurred (some rank/partition found >%d solutions).\n", maxres);
            printf("         Recap includes only the first %d per overflowing rank/partition.\n", maxres);
            if (partitions <= 64) {
                printf("Overflow locations:\n");
                for (int r = 0; r < world_size; r++) {
                    if (!masks || masks[r] == 0) continue;
                    for (int pidx = 0; pidx < partitions; pidx++) {
                        if (masks[r] & (UINT64_C(1) << pidx))
                            printf("  - rank %d, partition %d/%d\n", r, pidx + 1, partitions);
                    }
                }
            }
        }

        if (total > 0) {
            printf("Solutions (k1, k2) and origin:\n");
            for (int i = 0; i < total; i++) {
                const int part = (int)all[i].partition;
                if (partitions > 1)
                    printf("  - k1=%" PRIx64 ", k2=%" PRIx64 "  (rank %u, partition %d/%d)\n",
                           all[i].k1, all[i].k2, all[i].rank, part + 1, partitions);
                else
                    printf("  - k1=%" PRIx64 ", k2=%" PRIx64 "  (rank %u)\n",
                           all[i].k1, all[i].k2, all[i].rank);
            }
        }

        printf("=====================================================\n");
        fflush(stdout);
    }

    free(counts);
    free(displs);
    free(all);
    free(masks);
    free(anys);
    free(counts_b);
    free(displs_b);
}

// murmur64 hash functions, tailorized for 64-bit ints / Cf. Daniel Lemire
u64 murmur64(u64 x)
{
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;   
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}

/* represent n in 4 bytes */
void human_format(u64 n, char *target)
{
    if (n < 1000) {
        sprintf(target, "%" PRId64, n);
        return;
    }
    if (n < 1000000) {
        sprintf(target, "%.1fK", n / 1e3);
        return;
    }
    if (n < 1000000000) {
        sprintf(target, "%.1fM", n / 1e6);
        return;
    }
    if (n < 1000000000000ll) {
        sprintf(target, "%.1fG", n / 1e9);
        return;
    }
    if (n < 1000000000000000ll) {
        sprintf(target, "%.1fT", n / 1e12);
        return;
    }
}


/******************************** SPECK block cipher **************************/

#define ROTL32(x,r) (((x)<<(r)) | (x>>(32-(r))))
#define ROTR32(x,r) (((x)>>(r)) | ((x)<<(32-(r))))

#define ER32(x,y,k) (x=ROTR32(x,8), x+=y, x^=k, y=ROTL32(y,3), y^=x)
#define DR32(x,y,k) (y^=x, y=ROTR32(y,3), x^=k, x-=y, x=ROTL32(x,8))

void Speck64128KeySchedule(const u32 K[],u32 rk[])
{
    u32 i,D=K[3],C=K[2],B=K[1],A=K[0];
    for(i=0;i<27;){
        rk[i]=A; ER32(B,A,i++);
        rk[i]=A; ER32(C,A,i++);
        rk[i]=A; ER32(D,A,i++);
    }
}

void Speck64128Encrypt(const u32 Pt[], u32 Ct[], const u32 rk[])
{
    u32 i;
    Ct[0]=Pt[0]; Ct[1]=Pt[1];
    for(i=0;i<27;)
        ER32(Ct[1],Ct[0],rk[i++]);
}

void Speck64128Decrypt(u32 Pt[], const u32 Ct[], u32 const rk[])
{
    int i;
    Pt[0]=Ct[0]; Pt[1]=Ct[1];
    for(i=26;i>=0;)
        DR32(Pt[1],Pt[0],rk[i--]);
}


/******************************** dictionary ********************************/

/*
 * "classic" hash table for 64-bit key-value pairs, with linear probing.  
 * It operates under the assumption that the keys are somewhat random 64-bit integers.
 * The keys are only stored modulo 2**32 - 5 (a prime number), and this can lead 
 * to some false positives.
 */
static const u32 EMPTY = 0xffffffff;
static const u32 BUSY  = 0xfffffffe;
static const u64 PRIME = 0xfffffffb;

/* allocate a hash table with `size` slots (12*size bytes) */
void dict_setup(u64 size)
{
    int my_rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

	dict_size = size;
	// char hdsize[8];
	// human_format(dict_size * 12, hdsize);

	A_k = malloc(sizeof(*A_k) * dict_size);
	A_v = malloc(sizeof(*A_v) * dict_size);
	if (A_k == NULL || A_v == NULL)
		err(1, "impossible to allocate the dictionary for [%d]", my_rank);
	for (u64 i = 0; i < dict_size; i++)
		A_k[i] = EMPTY;
}

/* Reset dictionary content between partitions.
 * Important: do not free/malloc inside the partition loop. 
 */
void dict_reset(void)
{
    // Set every key to EMPTY (0xffffffff). Value bytes are irrelevant when k==EMPTY.
    memset(A_k, 0xff, (size_t)(dict_size * sizeof(*A_k)));
}

/* Lock-free insertion for concurrent fill (linear probing).
 * Claims an EMPTY slot using CAS, then publishes (v, k). 
 */
static inline void dict_insert_atomic(u64 key, u64 value)
{
    const u32 k = (u32)(key % PRIME);
    u64 h = murmur64(key) % dict_size;
    const u64 start_h = h;

    for (;;) {
        u32 cur = A_k[h];

        if (cur == EMPTY) {
            // Try to claim the slot.
            u32 prev = __sync_val_compare_and_swap(&A_k[h], EMPTY, BUSY);
            if (prev == EMPTY) {
                A_v[h] = value;
                __sync_synchronize();
                A_k[h] = k;
                return;
            }
            // Someone else raced us; retry this slot/probe.
            cur = prev;
        }

        // Occupied (including BUSY): probe next.
        h += 1;
        if (h == dict_size)
            h = 0;

        if (h == start_h) {
            fprintf(stderr, "CRITICAL ERROR: Full Hash Table! (Size: %" PRIu64 ")\n", dict_size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

/* Query the dictionnary with this `key`.  Write values (potentially) 
 *  matching the key in `values` and return their number. The `values`
 *  array must be preallocated of size (at least) `maxval`.
 *  The function returns -1 if there are more than `maxval` results.
 */
int dict_probe(u64 key, int maxval, u64 values[])
{
    u32 k = key % PRIME;
    u64 h = murmur64(key) % dict_size;
    int nval = 0;
    for (;;) {
        if (A_k[h] == EMPTY)
            return nval;
        if (A_k[h] == k) {
        	if (nval == maxval)
        		return -1;
            values[nval] = A_v[h];
            nval += 1;
        }
        h += 1;
        if (h == dict_size)
            h = 0;
   	}
}


/***************************** MITM problem ***********************************/

/* f : {0, 1}^n --> {0, 1}^n.  Speck64-128 encryption of P[0], using k */
u64 f(u64 k)
{
    assert((k & mask) == k);
    u32 K[4] = {k & 0xffffffff, k >> 32, 0, 0};
    u32 rk[27];
    Speck64128KeySchedule(K, rk);
    u32 Ct[2];
    Speck64128Encrypt(P[0], Ct, rk);
    return ((u64) Ct[0] ^ ((u64) Ct[1] << 32)) & mask;
}

/* g : {0, 1}^n --> {0, 1}^n.  speck64-128 decryption of C[0], using k */
u64 g(u64 k)
{
    assert((k & mask) == k);
    u32 K[4] = {k & 0xffffffff, k >> 32, 0, 0};
    u32 rk[27];
    Speck64128KeySchedule(K, rk);
    u32 Pt[2];
    Speck64128Decrypt(Pt, C[0], rk);
    return ((u64) Pt[0] ^ ((u64) Pt[1] << 32)) & mask;
}

bool is_good_pair(u64 k1, u64 k2)
{
    u32 Ka[4] = {k1 & 0xffffffff, k1 >> 32, 0, 0};
    u32 Kb[4] = {k2 & 0xffffffff, k2 >> 32, 0, 0};
    u32 rka[27];
    u32 rkb[27];
    Speck64128KeySchedule(Ka, rka);
    Speck64128KeySchedule(Kb, rkb);
    u32 mid[2];
    u32 Ct[2];
    Speck64128Encrypt(P[1], mid, rka);
    Speck64128Encrypt(mid, Ct, rkb);
    return (Ct[0] == C[1][0]) && (Ct[1] == C[1][1]);
}


/******************************************************************************/

static void drain_incoming(MPI_Request recv_reqs[RECV_SLOTS],
                           struct msg_entry recv_data[RECV_SLOTS][BUFFER_SIZE],
                           u64 *recv_inserts)
{
    int idx, flag;
    MPI_Status status;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    while (1) {
        MPI_Testany(RECV_SLOTS, recv_reqs, &idx, &flag, &status);

        if (!flag)
            break;

        int received_bytes;
        MPI_Get_count(&status, MPI_BYTE, &received_bytes);
        int n = received_bytes / (int)sizeof(struct msg_entry);

        for (int i = 0; i < n; i++) {
            dict_insert_atomic(recv_data[idx][i].z, recv_data[idx][i].x);
        }
		if (recv_inserts)
			*recv_inserts += (u64)n;

        // repost same slot
        MPI_Irecv(recv_data[idx],
                  sizeof(recv_data[idx]),
                  MPI_BYTE,
                  MPI_ANY_SOURCE,
                  TAG_FILL,
                  MPI_COMM_WORLD,
                  &recv_reqs[idx]);
    }
}

static void drain_incoming_probe(MPI_Request reqs[RECV_SLOTS],
                                    struct msg_entry data[RECV_SLOTS][BUFFER_SIZE],
                                    int maxres,
                                    u64 k1_out[],
                                    u64 k2_out[],
                                    int *nres,
                                    int *overflowed,
                                    u64 *work_recv,
                                    u64 *ncandidates)
{
    int idx, flag;
    MPI_Status st;

    // Thread-safe result recording: reserve a slot with atomic fetch-add.
    // We allow nres to exceed maxres; overflow is tracked separately.
    // (Helper is declared below, at file scope.)

    while (1) {
        MPI_Testany(RECV_SLOTS, reqs, &idx, &flag, &st);
        if (!flag) break;

        int bytes = 0;
        MPI_Get_count(&st, MPI_BYTE, &bytes);
        int nmsg = bytes / (int)sizeof(struct msg_entry);

        if (work_recv)
            *work_recv += (u64)nmsg;

        for (int m = 0; m < nmsg; m++) {
            u64 y  = data[idx][m].z; // query key
            u64 k2 = data[idx][m].x; // k2 to test

            u64 xs[4096];
            int nx = dict_probe(y, 4096, xs);

            if (nx == -1) {
                 // If we hit the limit, we might miss the solution.
                 // For now, just warn and process what we have (which is nothing, as dict_probe returns -1)
                 // Ideally dict_probe should return partial results or we should use a larger buffer.
                 // But with 4096 we should be safe.
                 fprintf(stderr, "WARNING: dict_probe overflow in drain_incoming_probe!\n");
            }

            if (ncandidates && nx > 0)
                *ncandidates += (u64)nx;

            for (int i = 0; i < nx; i++) {
                if (is_good_pair(xs[i], k2)) {
                    // reserve result slot atomically
                    int pos = __sync_fetch_and_add(nres, 1);
                    if (pos < maxres) {
                        k1_out[pos] = xs[i];
                        k2_out[pos] = k2;
                    } else {
                        __sync_fetch_and_or(overflowed, 1);
                    }
                }
            }
        }

        // repost same slot
        MPI_Irecv(data[idx], sizeof(data[idx]), MPI_BYTE,
                  MPI_ANY_SOURCE, TAG_PROBE_WORK, MPI_COMM_WORLD, &reqs[idx]);
    }
}

static void flush_buffer(struct send_slot *slot, int dest_rank,
                  MPI_Request recv_reqs[RECV_SLOTS],
                  struct msg_entry recv_data[RECV_SLOTS][BUFFER_SIZE],
				  u64 *recv_inserts)
{
    if (slot->buf.count == 0)
        return;

    while (slot->active) {
        int done;
        MPI_Test(&slot->req, &done, MPI_STATUS_IGNORE);
        if (done)
            slot->active = 0;
        else
            drain_incoming(recv_reqs, recv_data, recv_inserts);
    }

    MPI_Isend(slot->buf.data,
              slot->buf.count * sizeof(struct msg_entry),
              MPI_BYTE,
              dest_rank,
              TAG_FILL,
              MPI_COMM_WORLD,
              &slot->req);

    slot->active = 1;
    slot->buf.count = 0;
}

/* Flush a buffer using TAG_PROBE_WORK.
 * While waiting for a previous Isend to complete, we keep draining incoming probe work
 * so the rank continues to serve other ranks.
 */
static void flush_buffer_probe(struct send_slot *slot,
                               int dest_rank,
                               MPI_Request work_reqs[RECV_SLOTS],
                               struct msg_entry work_data[RECV_SLOTS][BUFFER_SIZE],
                               int maxres,
                               u64 k1_out[],
                               u64 k2_out[],
                               int *nres,
                               u64 *ncandidates,
                               int *overflowed,
                               u64 *work_recv)
{
    if (slot->buf.count == 0)
        return;

    while (slot->active) {
        int done = 0;
        MPI_Test(&slot->req, &done, MPI_STATUS_IGNORE);
        if (done)
            slot->active = 0;
        else
            drain_incoming_probe(work_reqs, work_data, maxres, k1_out, k2_out, nres, overflowed, work_recv, ncandidates);
    }

    MPI_Isend(slot->buf.data,
              slot->buf.count * sizeof(struct msg_entry),
              MPI_BYTE,
              dest_rank,
              TAG_PROBE_WORK,
              MPI_COMM_WORLD,
              &slot->req);

    slot->active = 1;
    slot->buf.count = 0;
}

static void add_to_buffer(struct send_slot *slot,
                          u64 z,
                          u64 x,
                          int dest,
                          MPI_Request recv_reqs[RECV_SLOTS],
                          struct msg_entry recv_data[RECV_SLOTS][BUFFER_SIZE],
					  u64 *recv_inserts)
{
    // Wait if the buffer is currently being sent (to avoid overwriting it)
    while (slot->active) {
        int done;
        MPI_Test(&slot->req, &done, MPI_STATUS_IGNORE);
        if (done)
            slot->active = 0;
        else
            drain_incoming(recv_reqs, recv_data, recv_inserts);
    }

    slot->buf.data[slot->buf.count].z = z;
    slot->buf.data[slot->buf.count].x = x;
    slot->buf.count++;

    if (slot->buf.count == BUFFER_SIZE) {
        flush_buffer(slot, dest, recv_reqs, recv_data, recv_inserts);
    }
}

static void add_to_buffer_probe(struct send_slot *slot,
                                u64 y,
                                u64 k2,
                                int dest_rank,
                                MPI_Request work_reqs[RECV_SLOTS],
                                struct msg_entry work_data[RECV_SLOTS][BUFFER_SIZE],
                                int maxres,
                                u64 k1_out[],
                                u64 k2_out[],
                                int *nres,
                                u64 *ncandidates,
                                int *overflowed,
                                u64 *work_recv)
{
    // Wait if the buffer is currently being sent
    while (slot->active) {
        int done = 0;
        MPI_Test(&slot->req, &done, MPI_STATUS_IGNORE);
        if (done)
            slot->active = 0;
        else
            drain_incoming_probe(work_reqs, work_data, maxres, k1_out, k2_out, nres, overflowed, work_recv, ncandidates);
    }

    slot->buf.data[slot->buf.count].z = y;
    slot->buf.data[slot->buf.count].x = k2;
    slot->buf.count += 1;

    if (slot->buf.count == BUFFER_SIZE) {
        flush_buffer_probe(slot, dest_rank, work_reqs, work_data,
                           maxres, k1_out, k2_out, nres, ncandidates, overflowed, work_recv);
    }
}

/* search the "golden collision" */
int golden_claw_search(int maxres, u64 k1[], u64 k2[], u64 keys_local,
                       int partition_idx, int total_partitions,
                       double *fill_seconds_out,
                       double *probe_seconds_out)
{  
    double start = wtime();
    int my_rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    
    u64 tot_keys = (1ull << n);
    u64 x_start = my_rank * keys_local;
    u64 x_end = x_start + keys_local;

    if (x_end > tot_keys) x_end = tot_keys; // Clamping
    if (x_start >= tot_keys) x_start = x_end = 0; // Handling eventual excess ranks
    
    struct send_slot *send_slots = malloc(p * sizeof(struct send_slot));
    for (int i = 0; i < p; i++) {
        send_slots[i].buf.count = 0;
        send_slots[i].active = 0;
    }

    MPI_Request recv_reqs[RECV_SLOTS];
    struct msg_entry recv_data[RECV_SLOTS][BUFFER_SIZE];
    u64 recv_inserts = 0;
    for (int i = 0; i < RECV_SLOTS; i++) {
        MPI_Irecv(recv_data[i],
                  sizeof(recv_data[i]),
                  MPI_BYTE,
                  MPI_ANY_SOURCE,
                  TAG_FILL,
                  MPI_COMM_WORLD,
                  &recv_reqs[i]);
    }

    u64 local_sent_records = 0;
    
    // --- PHASE 1: FILL LOOP ---
    enum { LOCAL_BATCH = 256 };
    struct local_item { u64 z; u64 x; int owner; };

    // Interleaved slicing: distribute work for every partition across all ranks.
    // Requirement: total_partitions must be a power of two (validated in main).
    const u64 part_mask = (u64)(total_partitions - 1);

    // Heartbeat counters (progress over the scanned local key range)
    const u64 fill_total = (x_end > x_start) ? (x_end - x_start) : 0;
    volatile u64 fill_seen = 0;
    int next_fill_pct = 10; // print every 10%

#ifdef _OPENMP
#pragma omp parallel reduction(+:local_sent_records)
#endif
    {
        struct local_item local_buf[LOCAL_BATCH];
        int local_count = 0;

        u64 seen_local = 0;

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (u64 x = x_start; x < x_end; x++) {
            // Heartbeat progress on scanned x (not only inserted x)
            seen_local += 1;
            if ((seen_local & 0x3fffull) == 0) { // amortize atomics (~every 16k iters)
                __sync_fetch_and_add((u64 *)&fill_seen, seen_local);
                seen_local = 0;
            }

#ifdef _OPENMP
            if (verbose && my_rank == 0 && omp_get_thread_num() == 0 && fill_total) {
#else
            if (verbose && my_rank == 0 && fill_total) {
#endif
                u64 cur = __sync_fetch_and_add((u64 *)&fill_seen, 0);
                int pct = (int)((cur * 100ull) / fill_total);
                if (pct >= next_fill_pct) {
                    printf("[HEARTBEAT] Partition %d/%d - Phase: FILL - Progress: %d%%\n",
                           partition_idx + 1, total_partitions, next_fill_pct);
                    fflush(stdout);
                    while (next_fill_pct <= pct)
                        next_fill_pct += 10;
                }
            }

            if (total_partitions > 1) {
                if ((x & part_mask) != (u64)partition_idx)
                    continue;
            }
            u64 z = f(x);
            int owner = (int)(murmur64(z) % (u64)p);

            local_buf[local_count].z = z;
            local_buf[local_count].x = x;
            local_buf[local_count].owner = owner;
            local_count++;

            if (local_count == LOCAL_BATCH) {
                // 1) Local inserts (lock-free)
                for (int i = 0; i < local_count; i++) {
                    if (local_buf[i].owner == my_rank) {
                        dict_insert_atomic(local_buf[i].z, local_buf[i].x);
                    }
                }

                // 2) Remote sends + 3) periodic drain_incoming (serialized MPI)
#ifdef _OPENMP
#pragma omp critical(mpi_send)
#endif
                {
                    for (int i = 0; i < local_count; i++) {
                        if (local_buf[i].owner != my_rank) {
                            local_sent_records += 1;
                            add_to_buffer(&send_slots[local_buf[i].owner],
                                          local_buf[i].z,
                                          local_buf[i].x,
                                          local_buf[i].owner,
                                          recv_reqs,
                                          recv_data,
                                          &recv_inserts);
                        }
                    }
                    drain_incoming(recv_reqs, recv_data, &recv_inserts);
                }

                local_count = 0;
            }
        }

        // Flush any remaining items from this thread
        if (local_count) {
            for (int i = 0; i < local_count; i++) {
                if (local_buf[i].owner == my_rank) {
                    dict_insert_atomic(local_buf[i].z, local_buf[i].x);
                }
            }

#ifdef _OPENMP
#pragma omp critical(mpi_send)
#endif
            {
                for (int i = 0; i < local_count; i++) {
                    if (local_buf[i].owner != my_rank) {
                        local_sent_records += 1;
                        add_to_buffer(&send_slots[local_buf[i].owner],
                                      local_buf[i].z,
                                      local_buf[i].x,
                                      local_buf[i].owner,
                                      recv_reqs,
                                      recv_data,
                                      &recv_inserts);
                    }
                }
                drain_incoming(recv_reqs, recv_data, &recv_inserts);
            }
        }

        if (seen_local)
            __sync_fetch_and_add((u64 *)&fill_seen, seen_local);
    }

    // Final heartbeat for fill
    if (verbose && my_rank == 0 && fill_total) {
        __sync_fetch_and_add((u64 *)&fill_seen, 0); // ensure visibility
        printf("[HEARTBEAT] Partition %d/%d - Phase: FILL - Progress: 100%%\n",
               partition_idx + 1, total_partitions);
        fflush(stdout);
    }

    // --- FLUSH LEFTOVERS ---
    for (int i = 0; i < p; i++) {
        if (i != my_rank) {
            flush_buffer(&send_slots[i], i, recv_reqs, recv_data, &recv_inserts);
        }
    }

    int pending = 1;
    while (pending) {
        pending = 0;

        for (int i = 0; i < p; i++) {
            if (send_slots[i].active) {
                int done;
                MPI_Test(&send_slots[i].req, &done, MPI_STATUS_IGNORE);
                if (done)
                    send_slots[i].active = 0;
                else
                    pending = 1;
            }
        }

        drain_incoming(recv_reqs, recv_data, &recv_inserts);
    }

    MPI_Request barrier_req;
    MPI_Ibarrier(MPI_COMM_WORLD, &barrier_req);

    int barrier_done = 0;
    while (!barrier_done) {
        drain_incoming(recv_reqs, recv_data, &recv_inserts);
        MPI_Test(&barrier_req, &barrier_done, MPI_STATUS_IGNORE);
    }

    // IMPORTANT: MPI_Ibarrier completion does NOT imply all TAG_FILL messages have been
    // matched/received. Avoid cancelling Irecvs too early (can cause lost inserts -> false negatives).
    // Wait until every remotely-sent record has been received and inserted.
    {
        u64 global_sent = 0;
        MPI_Allreduce(&local_sent_records, &global_sent, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        for (;;) {
            drain_incoming(recv_reqs, recv_data, &recv_inserts);
            u64 global_recv = 0;
            MPI_Allreduce(&recv_inserts, &global_recv, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
            if (global_recv == global_sent) {
                break;
            }
        }
    }

    // --- CLEANUP RECVS ---
    drain_incoming(recv_reqs, recv_data, &recv_inserts);
    for (int i = 0; i < RECV_SLOTS; i++) {
        MPI_Status st;
        int was_cancelled = 0;

        MPI_Cancel(&recv_reqs[i]);
        MPI_Wait(&recv_reqs[i], &st);
        MPI_Test_cancelled(&st, &was_cancelled);

        if (!was_cancelled) {
            int received_bytes = 0;
            MPI_Get_count(&st, MPI_BYTE, &received_bytes);
            int nmsg = received_bytes / (int)sizeof(struct msg_entry);
            for (int j = 0; j < nmsg; j++) {
                dict_insert_atomic(recv_data[i][j].z, recv_data[i][j].x);
            }
            recv_inserts += (u64)nmsg;
        }
    }

    // timing: always print (rank 0 only unless --verbose)
    MPI_Barrier(MPI_COMM_WORLD);
    double fill_seconds = wtime() - start;
    if (fill_seconds_out)
        *fill_seconds_out = fill_seconds;
    if (verbose || my_rank == 0) {
        printf("[%d] Fill completed in %.3f seconds (partition %d/%d, x_start=%" PRIu64 ", x_end=%" PRIu64 ")\n",
               my_rank, fill_seconds, partition_idx+1, total_partitions, x_start, x_end);
        fflush(stdout);
    }

    free(send_slots);

    
    // --- PHASE 2: PROBE ---
    MPI_Barrier(MPI_COMM_WORLD);
    double mid = wtime();

    struct send_slot *probe_slots = malloc(p * sizeof(struct send_slot));
    if (probe_slots == NULL)
        err(1, "impossible to allocate probe send slots for [%d]", my_rank);

    for (int i = 0; i < p; i++) {
        probe_slots[i].buf.count = 0;
        probe_slots[i].active = 0;
    }

    MPI_Request work_reqs[RECV_SLOTS];
    struct msg_entry work_data[RECV_SLOTS][BUFFER_SIZE];

    for (int i = 0; i < RECV_SLOTS; i++) {
        MPI_Irecv(work_data[i],
                  sizeof(work_data[i]),
                  MPI_BYTE,
                  MPI_ANY_SOURCE,
                  TAG_PROBE_WORK,
                  MPI_COMM_WORLD,
                  &work_reqs[i]);
    }

    u64 k2_start = my_rank * keys_local;
    u64 k2_end = k2_start + keys_local;
    if (k2_end > tot_keys) k2_end = tot_keys;
    if (k2_start >= tot_keys) k2_start = k2_end = 0;

    int nres = 0;
    u64 ncandidates = 0;
    int overflowed = 0;

    // Track remote work messages to ensure we never cancel receives too early.
    u64 probe_sent_records = 0;
    u64 probe_recv_records = 0;

    // Heartbeat counters (progress over scanned k2 range)
    const u64 probe_total = (k2_end > k2_start) ? (k2_end - k2_start) : 0;
    volatile u64 probe_seen = 0;
    int next_probe_pct = 10; // print every 10%

    enum { LOCAL_BATCH2 = 256 };
    struct probe_item { u64 y; u64 k2; int owner; };

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        struct probe_item local_buf[LOCAL_BATCH2];
        int local_count = 0;

        u64 seen_local = 0;

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (u64 k2_val = k2_start; k2_val < k2_end; k2_val++) {
            // Heartbeat progress on scanned k2
            seen_local += 1;
            if ((seen_local & 0x3fffull) == 0) { // amortize atomics (~every 16k iters)
                __sync_fetch_and_add((u64 *)&probe_seen, seen_local);
                seen_local = 0;
            }

#ifdef _OPENMP
            if (verbose && my_rank == 0 && omp_get_thread_num() == 0 && probe_total) {
#else
            if (verbose && my_rank == 0 && probe_total) {
#endif
                u64 cur = __sync_fetch_and_add((u64 *)&probe_seen, 0);
                int pct = (int)((cur * 100ull) / probe_total);
                if (pct >= next_probe_pct) {
                    printf("[HEARTBEAT] Partition %d/%d - Phase: PROBE - Progress: %d%%\n",
                           partition_idx + 1, total_partitions, next_probe_pct);
                    fflush(stdout);
                    while (next_probe_pct <= pct)
                        next_probe_pct += 10;
                }
            }

            u64 y = g(k2_val);
            int owner = (int)(murmur64(y) % (u64)p);

            local_buf[local_count].y = y;
            local_buf[local_count].k2 = k2_val;
            local_buf[local_count].owner = owner;
            local_count++;

            if (local_count == LOCAL_BATCH2) {
                // Local probes (no MPI)
                for (int i = 0; i < local_count; i++) {
                    if (local_buf[i].owner == my_rank) {
                        u64 xs[4096];
                        int nx = dict_probe(local_buf[i].y, 4096, xs);
                        if (nx < 0) {
                             fprintf(stderr, "WARNING: dict_probe overflow in local probe!\n");
                             nx = 0;
                        }
                        __sync_fetch_and_add(&ncandidates, (u64)nx);

                        if (!__sync_fetch_and_add(&overflowed, 0)) {
                            for (int j = 0; j < nx; j++) {
                                if (is_good_pair(xs[j], local_buf[i].k2)) {
                                    int pos = __sync_fetch_and_add(&nres, 1);
                                    if (pos < maxres) {
                                        k1[pos] = xs[j];
                                        k2[pos] = local_buf[i].k2;
                                    } else {
                                        __sync_fetch_and_or(&overflowed, 1);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

                // Remote sends + periodic drain_incoming_probe (serialized MPI)
#ifdef _OPENMP
#pragma omp critical(mpi_send)
#endif
                {
                    for (int i = 0; i < local_count; i++) {
                        if (local_buf[i].owner != my_rank) {
                            probe_sent_records += 1;
                            add_to_buffer_probe(&probe_slots[local_buf[i].owner],
                                                local_buf[i].y,
                                                local_buf[i].k2,
                                                local_buf[i].owner,
                                                work_reqs,
                                                work_data,
                                                maxres,
                                                k1,
                                                k2,
                                                &nres,
                                                &ncandidates,
                                                &overflowed,
                                                &probe_recv_records);
                        }
                    }
                    drain_incoming_probe(work_reqs, work_data, maxres, k1, k2, &nres, &overflowed, &probe_recv_records, &ncandidates);
                }

                local_count = 0;
            }
        }

        // Flush any remaining items from this thread
        if (local_count) {
            for (int i = 0; i < local_count; i++) {
                if (local_buf[i].owner == my_rank) {
                    u64 xs[4096];
                    int nx = dict_probe(local_buf[i].y, 4096, xs);
                    if (nx < 0) {
                         fprintf(stderr, "WARNING: dict_probe overflow in local probe (flush)!\n");
                         nx = 0;
                    }
                    __sync_fetch_and_add(&ncandidates, (u64)nx);

                    if (!__sync_fetch_and_add(&overflowed, 0)) {
                        for (int j = 0; j < nx; j++) {
                            if (is_good_pair(xs[j], local_buf[i].k2)) {
                                int pos = __sync_fetch_and_add(&nres, 1);
                                if (pos < maxres) {
                                    k1[pos] = xs[j];
                                    k2[pos] = local_buf[i].k2;
                                } else {
                                    __sync_fetch_and_or(&overflowed, 1);
                                    break;
                                }
                            }
                        }
                    }
                }
            }

#ifdef _OPENMP
#pragma omp critical(mpi_send)
#endif
            {
                for (int i = 0; i < local_count; i++) {
                    if (local_buf[i].owner != my_rank) {
                        probe_sent_records += 1;
                        add_to_buffer_probe(&probe_slots[local_buf[i].owner],
                                            local_buf[i].y,
                                            local_buf[i].k2,
                                            local_buf[i].owner,
                                            work_reqs,
                                            work_data,
                                            maxres,
                                            k1,
                                            k2,
                                            &nres,
                                            &ncandidates,
                                            &overflowed,
                                            &probe_recv_records);
                    }
                }
                drain_incoming_probe(work_reqs, work_data, maxres, k1, k2, &nres, &overflowed, &probe_recv_records, &ncandidates);
            }
        }

        if (seen_local)
            __sync_fetch_and_add((u64 *)&probe_seen, seen_local);
    }

    // Final heartbeat for probe
    if (verbose && my_rank == 0 && probe_total) {
        __sync_fetch_and_add((u64 *)&probe_seen, 0); // ensure visibility
        printf("[HEARTBEAT] Partition %d/%d - Phase: PROBE - Progress: 100%%\n",
               partition_idx + 1, total_partitions);
        fflush(stdout);
    }

    // Flush remaining buffered work items
    for (int i = 0; i < p; i++) {
        if (i != my_rank) {
            flush_buffer_probe(&probe_slots[i],
                               i,
                               work_reqs,
                               work_data,
                               maxres,
                               k1,
                               k2,
                               &nres,
                               &ncandidates,
                               &overflowed,
                               &probe_recv_records);
        }
    }

    // Wait for all Isends to complete, while continuing to serve incoming work
    int pending_probe = 1;
    while (pending_probe) {
        pending_probe = 0;
        for (int i = 0; i < p; i++) {
            if (probe_slots[i].active) {
                int done = 0;
                MPI_Test(&probe_slots[i].req, &done, MPI_STATUS_IGNORE);
                if (done)
                    probe_slots[i].active = 0;
                else
                    pending_probe = 1;
            }
        }
        drain_incoming_probe(work_reqs, work_data, maxres, k1, k2, &nres, &overflowed, &probe_recv_records, &ncandidates);
    }

    // Non-blocking barrier: keep serving work until everybody is done generating it
    MPI_Request barrier_req2;
    MPI_Ibarrier(MPI_COMM_WORLD, &barrier_req2);

    int barrier_done2 = 0;
    while (!barrier_done2) {
        drain_incoming_probe(work_reqs, work_data, maxres, k1, k2, &nres, &overflowed, &probe_recv_records, &ncandidates);
        MPI_Test(&barrier_req2, &barrier_done2, MPI_STATUS_IGNORE);
    }

    // Like in fill: after all ranks finished generating work (barrier complete),
    // ensure every remotely-sent work record has been received and processed before cancelling Irecvs.
    {
        u64 global_sent = 0;
        MPI_Allreduce(&probe_sent_records, &global_sent, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        for (;;) {
            drain_incoming_probe(work_reqs, work_data, maxres, k1, k2, &nres, &overflowed, &probe_recv_records, &ncandidates);
            u64 global_recv = 0;
            MPI_Allreduce(&probe_recv_records, &global_recv, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
            if (global_recv == global_sent)
                break;
        }
    }

    // Final drain + cancellation of posted Irecvs (avoid leaving pending requests)
    drain_incoming_probe(work_reqs, work_data, maxres, k1, k2, &nres, &overflowed, &probe_recv_records, &ncandidates);

    for (int i = 0; i < RECV_SLOTS; i++) {
        MPI_Status st;
        int was_cancelled = 0;

        MPI_Cancel(&work_reqs[i]);
        MPI_Wait(&work_reqs[i], &st);
        MPI_Test_cancelled(&st, &was_cancelled);

        if (!was_cancelled) {
            int received_bytes = 0;
            MPI_Get_count(&st, MPI_BYTE, &received_bytes);
            int nmsg = received_bytes / (int)sizeof(struct msg_entry);

            for (int m = 0; m < nmsg; m++) {
                u64 y = work_data[i][m].z;
                u64 k2_val = work_data[i][m].x;

                u64 xs[256];
                int nx = dict_probe(y, 256, xs);
                assert(nx >= 0);
                ncandidates += (u64)nx;

                if (!overflowed) {
                    for (int j = 0; j < nx; j++) {
                        if (is_good_pair(xs[j], k2_val)) {
                            int pos = __sync_fetch_and_add(&nres, 1);
                            if (pos < maxres) {
                                k1[pos] = xs[j];
                                k2[pos] = k2_val;
                            } else {
                                __sync_fetch_and_or(&overflowed, 1);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // timing: always print (rank 0 only unless --verbose)
    double probe_seconds = wtime() - mid;
    if (probe_seconds_out)
        *probe_seconds_out = probe_seconds;
    if (verbose || my_rank == 0) {
        printf("[%d] Probe completed in %.3f seconds (partition %d/%d, k2_start=%" PRIu64 ", k2_end=%" PRIu64 "), candidates=%" PRIu64 ", nres=%d\n",
               my_rank, probe_seconds, partition_idx+1, total_partitions, k2_start, k2_end, ncandidates, nres);
        fflush(stdout);
    }

    free(probe_slots);

    if (overflowed)
        return -1;

    return nres;
}


/************************** command-line options ****************************/

void usage(char **argv)
{
        printf("%s [OPTIONS]\n\n", argv[0]);
        printf("Options:\n");
        printf("--n N                       block size [default 24]\n");
        printf("--C0 N                      1st ciphertext (in hex)\n");
        printf("--C1 N                      2nd ciphertext (in hex)\n");
		printf("--partitions P              enable slicing with P partitions (power of two). Default: disabled\n");
		printf("--verbose                   enable verbose progress/diagnostic output\n");
        printf("\n");
        printf("--n, --C0, --C1 are required\n");
        exit(0);
}

void process_command_line_options(int argc, char ** argv)
{
        struct option longopts[6] = {
                {"n", required_argument, NULL, 'n'},
                {"C0", required_argument, NULL, '0'},
                {"C1", required_argument, NULL, '1'},
                {"partitions", required_argument, NULL, 'p'},
                {"verbose", no_argument, NULL, 'V'},
                {NULL, 0, NULL, 0}
        };
        char ch;
        int set = 0;
        while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
                switch (ch) {
                case 'n':
                        n = atoi(optarg);
                        mask = (1ull << n) - 1;
                        break;
                case 'V':
                        verbose = 1;
                        break;
                case 'p': {
                        int v = atoi(optarg);
                        // If user passes 0 or 1 -> treat as disabled.
                        if (v <= 1) {
                            num_partitions = 1;
                        } else {
                            num_partitions = v;
                        }
                        break;
                }
                case '0':
                        set |= 1;
                        u64 c0 = strtoull(optarg, NULL, 16);
                        C[0][0] = c0 & 0xffffffff;
                        C[0][1] = c0 >> 32;
                        break;
                case '1':
                        set |= 2;
                        u64 c1 = strtoull(optarg, NULL, 16);
                        C[1][0] = c1 & 0xffffffff;
                        C[1][1] = c1 >> 32;
                        break;
                default:
                        errx(1, "Unknown option\n");
                }
        }
        if (n == 0 || set != 3) {
        	usage(argv);
        	exit(1);
        }
}

/******************************************************************************/

int main(int argc, char **argv)
{
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "MPI does not provide required thread level: need MULTIPLE, got %d\n", provided);
        // MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int my_rank, p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	process_command_line_options(argc, argv);
    
    // Local key-space chunk for each process (ceil division)
    u64 keys_local = ((1ull << n) + p - 1) / p;
 
    // Avoids printing it p times
    if (verbose && my_rank == 0){
        printf("Running with n=%d on %d MPI processes, C0=(%08x, %08x) and C1=(%08x, %08x)\n", 
            (int) n, p, C[0][0], C[0][1], C[1][0], C[1][1]);
        printf("Search space: 2^%d = %" PRIu64 " keys\n", (int)n, (u64)(UINT64_C(1) << n));
        printf("Keys per core: %" PRIu64 "\n", keys_local);
    }
    
    // Slicing parameter (time-memory tradeoff). Default: disabled unless user sets --partitions.
    int partitions = num_partitions;
    if (partitions <= 1) {
        partitions = 1;
    }
    if ((partitions & (partitions - 1)) != 0) {
        if (my_rank == 0)
            fprintf(stderr, "--partitions must be a power of two (got %d)\n", partitions);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Hash-table capacity per partition: size for ~keys_local/partitions elements (+ padding).
    // Use ceil-div to avoid under-allocating for small sizes.
    u64 keys_local_slice = (keys_local + (u64)partitions - 1) / (u64)partitions;

    // Adaptive padding
    u64 alloc_size;
    if (n >= 32) {
        alloc_size = keys_local_slice * 2;
        if (verbose && my_rank == 0) printf(">> Mode: FAST (n>=32). Padding: 2.0x\n");
    } else if (n < 20) {
        alloc_size = (keys_local_slice * 3) / 2;
        if (verbose && my_rank == 0) printf(">> Mode: SAFE (n<20). Padding: 1.5x\n");
    } else {
        alloc_size = (keys_local_slice * 9) / 8;
        if (verbose && my_rank == 0) printf(">> Mode: LEAN (20<=n<32). Padding: 1.125x\n");
    }

    // SAFETY FLOOR for very small n (avoid too small hash tables)
    if (alloc_size < 4096) {
        alloc_size = 4096;
    }

    if (verbose && my_rank == 0) {
        if (partitions > 1)
            printf(">> Slicing enabled: %d partitions\n", partitions);
        else
            printf(">> Slicing disabled (normal run)\n");
        printf(">> Per-rank dictionary slots: %" PRIu64 " (keys_local_slice=%" PRIu64 ")\n",
               alloc_size, keys_local_slice);
    }

    // Initialize dictionary once, then clear it between partitions.
    dict_setup(alloc_size);

    /* search */
    const int maxres = 16;
    u64 k1[16], k2[16]; // per-rank result buffers (maxres=16)

    // Per-rank solution log for the final recap.
    const size_t local_cap = (size_t)maxres * (size_t)partitions;
    struct solution_rec *local_solutions = (struct solution_rec *)calloc(local_cap, sizeof(struct solution_rec));
    if (local_solutions == NULL)
        err(1, "impossible to allocate local solution log for [%d]", my_rank);
    int local_n = 0;
    u64 overflow_mask = 0;
    int overflow_any = 0;

    struct part_timing *timings = NULL;
    if (my_rank == 0) {
        timings = (struct part_timing *)calloc((size_t)partitions, sizeof(struct part_timing));
        if (timings == NULL)
            err(1, "impossible to allocate timings table");
    }
    double total_effective_time = 0.0;

    if (partitions == 1) {
        double fill_s = 0.0, probe_s = 0.0;
        int nkey_raw = golden_claw_search(maxres, k1, k2, keys_local, 0, 1, &fill_s, &probe_s);
        int overflowed = (nkey_raw < 0);
        int nkey = overflowed ? maxres : nkey_raw;

        if (overflowed) {
            overflow_mask |= UINT64_C(1);
        }

        // Save locally for the recap.
        for (int i = 0; i < nkey && (size_t)local_n < local_cap; i++) {
            local_solutions[local_n].k1 = k1[i];
            local_solutions[local_n].k2 = k2[i];
            local_solutions[local_n].rank = (u32i)my_rank;
            local_solutions[local_n].partition = 0;
            local_n += 1;
        }

        int total_solutions = 0;
        MPI_Allreduce(&nkey, &total_solutions, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        // Reduce timing stats for the single pass.
        double fill_min = 0.0, fill_max = 0.0, fill_sum = 0.0;
        double probe_min = 0.0, probe_max = 0.0, probe_sum = 0.0;
        MPI_Reduce(&fill_s, &fill_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&fill_s, &fill_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&fill_s, &fill_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&probe_s, &probe_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&probe_s, &probe_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&probe_s, &probe_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (my_rank == 0) {
            timings[0].fill_min = fill_min;
            timings[0].fill_max = fill_max;
            timings[0].fill_avg = fill_sum / (double)p;
            timings[0].probe_min = probe_min;
            timings[0].probe_max = probe_max;
            timings[0].probe_avg = probe_sum / (double)p;
            total_effective_time = fill_max + probe_max;
        }

        if (my_rank == 0) {
            printf("[TOTAL] solutions across all ranks: %d\n", total_solutions);
            fflush(stdout);
        }

        if (nkey > 0) {
            for (int i = 0; i < nkey; i++) {
                assert(f(k1[i]) == g(k2[i]));
                assert(is_good_pair(k1[i], k2[i]));
                printf("[rank %d] Solution found: (%" PRIx64 ", %" PRIx64 ") [checked OK]\n",
                       my_rank, k1[i], k2[i]);
                fflush(stdout);
            }
        }
    } else {
        int total_solutions_all_partitions = 0;
        for (int pidx = 0; pidx < partitions; pidx++) {
            // Synchronize between partitions to keep phases aligned across ranks.
            MPI_Barrier(MPI_COMM_WORLD);
            dict_reset();
            MPI_Barrier(MPI_COMM_WORLD);

            double fill_s = 0.0, probe_s = 0.0;
            int nkey_raw = golden_claw_search(maxres, k1, k2, keys_local, pidx, partitions, &fill_s, &probe_s);
            int overflowed = (nkey_raw < 0);
            int nkey = overflowed ? maxres : nkey_raw;

            if (overflowed) {
                if (partitions <= 64)
                    overflow_mask |= (UINT64_C(1) << pidx);
                else
                    overflow_any = 1;
            }

            // Save locally for the recap.
            for (int i = 0; i < nkey && (size_t)local_n < local_cap; i++) {
                local_solutions[local_n].k1 = k1[i];
                local_solutions[local_n].k2 = k2[i];
                local_solutions[local_n].rank = (u32i)my_rank;
                local_solutions[local_n].partition = (u32i)pidx;
                local_n += 1;
            }

            int total_solutions_partition = 0;
            MPI_Allreduce(&nkey, &total_solutions_partition, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);

            // Reduce timing stats for this partition.
            double fill_min = 0.0, fill_max = 0.0, fill_sum = 0.0;
            double probe_min = 0.0, probe_max = 0.0, probe_sum = 0.0;
            MPI_Reduce(&fill_s, &fill_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(&fill_s, &fill_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&fill_s, &fill_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&probe_s, &probe_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(&probe_s, &probe_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&probe_s, &probe_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (my_rank == 0) {
                timings[pidx].fill_min = fill_min;
                timings[pidx].fill_max = fill_max;
                timings[pidx].fill_avg = fill_sum / (double)p;
                timings[pidx].probe_min = probe_min;
                timings[pidx].probe_max = probe_max;
                timings[pidx].probe_avg = probe_sum / (double)p;
                total_effective_time += (fill_max + probe_max);
            }

            if (my_rank == 0) {
                printf("[TOTAL][partition %d/%d] solutions across all ranks: %d\n",
                       pidx+1, partitions, total_solutions_partition);
                fflush(stdout);
            }

            if (nkey > 0) {
                for (int i = 0; i < nkey; i++) {
                    assert(f(k1[i]) == g(k2[i]));
                    assert(is_good_pair(k1[i], k2[i]));
                    printf("[rank %d] Solution found (partition %d/%d): (%" PRIx64 ", %" PRIx64 ") [checked OK]\n",
                           my_rank, pidx, partitions, k1[i], k2[i]);
                    fflush(stdout);
                }
            }

            total_solutions_all_partitions += total_solutions_partition;
        }

        if (my_rank == 0) {
            printf("[TOTAL][all partitions] solutions across all ranks (sum of per-partition totals): %d\n",
                   total_solutions_all_partitions);
            fflush(stdout);
        }
    }

    // Final recap: consolidate found solutions across all ranks.
    MPI_Barrier(MPI_COMM_WORLD);
    final_recap_print(my_rank, p, partitions, maxres,
                      local_solutions, local_n,
                      (partitions <= 64) ? overflow_mask : 0,
                      (partitions > 64) ? overflow_any : 0,
                      timings,
                      total_effective_time);

    if (my_rank == 0)
        free(timings);

    free(local_solutions);

    MPI_Finalize();
    return 0;
}