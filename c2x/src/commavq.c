// commavq.c

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <time.h>
#include <errno.h>
#include <zlib.h>

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#include <direct.h>
#define PATH_SEPARATOR "\\"
typedef HANDLE thread_t;
typedef CRITICAL_SECTION mutex_t;
typedef CONDITION_VARIABLE cond_t;
#define THREAD_RETURN unsigned int __stdcall
#define THREAD_RETURN_TYPE unsigned int
#else
#include <pthread.h>
#include <unistd.h>
#include <sys/wait.h>
#define PATH_SEPARATOR "/"
typedef pthread_t thread_t;
typedef pthread_mutex_t mutex_t;
typedef pthread_cond_t cond_t;
#define THREAD_RETURN void *
#define THREAD_RETURN_TYPE void *
#endif

#define VERSION "v0.3-opbitstream"
#define MAX_FILENAME 512

// Constants based on file format
#define HEADER_SIZE 128
#define TOKENS_PER_FRAME 128 // 8x16
#define BITS_PER_TOKEN 10
#define FRAME_SIZE 256                                            // Original unpacked bytes
#define PACKED_FRAME_SIZE (TOKENS_PER_FRAME * BITS_PER_TOKEN / 8) // 160
#define NUM_FRAMES 1200
#define TOTAL_DATA_SIZE (NUM_FRAMES * FRAME_SIZE)               // 307200
#define PACKED_TOTAL_DATA_SIZE (NUM_FRAMES * PACKED_FRAME_SIZE) // 192000
#define TOTAL_FILE_SIZE (HEADER_SIZE + TOTAL_DATA_SIZE)
#define MASK_SIZE (TOKENS_PER_FRAME / 8) // 16

// New advanced encoding modes (control byte has MSB=1)
#define CONTROL_FLAG 0x80         // MSB indicates special frame
#define CONTROL_SUBTYPE_MASK 0x60 // bits 6-5
#define CONTROL_SUBTYPE_SHIFT 5
#define CONTROL_SUBTYPE_SPARSE 0x00 // bits 6-5 = 00 -> sparse changed tokens
#define CONTROL_SUBTYPE_RLE 0x20    // bits 6-5 = 01 -> run of identical frames

// Lower 5 bits meaning depend on subtype:
//  SPARSE: number_of_changed_tokens_minus_1 (1..32 tokens) => supports up to 32 changes in sparse frame
//  RLE:    run_length_minus_1 (1..32 identical frames)

// Threshold for using sparse mode instead of dense (mask) mode
#define SPARSE_THRESHOLD 2

// Helper macro
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Unified opcode bitstream modes (2 bits each)
#define OPCODE_DENSE 0u  // 00: full mask + values
#define OPCODE_SPARSE 1u // 01: small count (<=32) indices + values
#define OPCODE_RLE 2u    // 10: repeat previous frame run (varint run-1)
// 11 reserved

typedef struct
{
    uint8_t *buf;
    size_t capacity; // allocated bytes
    size_t byte_pos; // current byte being written
    int bit_pos;     // next bit index in current byte 0..7
} bit_writer_t;

typedef struct
{
    const uint8_t *buf;
    size_t size_bytes;
    size_t byte_pos;
    int bit_pos;         // next bit to read in current byte
    uint32_t total_bits; // total valid bits in stream
    uint32_t bits_read;  // bits consumed
} bit_reader_t;

static void bw_init(bit_writer_t *bw, size_t initial_cap)
{
    bw->buf = (uint8_t *)malloc(initial_cap);
    bw->capacity = initial_cap;
    bw->byte_pos = 0;
    bw->bit_pos = 0;
    if (bw->buf)
        memset(bw->buf, 0, initial_cap);
}
static void bw_reserve(bit_writer_t *bw, size_t need_bits)
{
    size_t need_bytes = (need_bits + 7) / 8;
    if (need_bytes <= bw->capacity)
        return;
    size_t newcap = bw->capacity ? bw->capacity * 2 : 1024;
    if (newcap < need_bytes)
        newcap = need_bytes;
    uint8_t *n = (uint8_t *)realloc(bw->buf, newcap);
    if (n)
    {
        memset(n + bw->capacity, 0, newcap - bw->capacity);
        bw->buf = n;
        bw->capacity = newcap;
    }
}
static void bw_put_bits(bit_writer_t *bw, uint32_t value, int bits)
{
    if (bits <= 0)
        return;
    size_t start_bit = bw->byte_pos * 8 + bw->bit_pos;
    bw_reserve(bw, start_bit + bits + 8);
    for (int i = 0; i < bits; i++)
    {
        int bit = (value >> i) & 1; // LSB-first insertion
        bw->buf[bw->byte_pos] |= (bit & 1) << bw->bit_pos;
        bw->bit_pos++;
        if (bw->bit_pos == 8)
        {
            bw->bit_pos = 0;
            bw->byte_pos++;
        }
    }
}
static void bw_put_varint(bit_writer_t *bw, uint32_t v)
{
    // Standard 7-bit groups little-endian, continuation bit in MSB of each byte
    do
    {
        uint8_t part = v & 0x7F;
        v >>= 7;
        if (v)
            part |= 0x80;
        bw_put_bits(bw, part, 8);
    } while (v);
}
static uint32_t bw_finish(const bit_writer_t *bw)
{
    return (uint32_t)(bw->byte_pos * 8 + bw->bit_pos);
}
static void bw_free(bit_writer_t *bw)
{
    if (bw->buf)
    {
        free(bw->buf);
        bw->buf = NULL;
    }
}

static void br_init(bit_reader_t *br, const uint8_t *buf, size_t size_bytes, uint32_t total_bits)
{
    br->buf = buf;
    br->size_bytes = size_bytes;
    br->byte_pos = 0;
    br->bit_pos = 0;
    br->total_bits = total_bits;
    br->bits_read = 0;
}
static uint32_t br_get_bits(bit_reader_t *br, int bits)
{
    uint32_t v = 0;
    for (int i = 0; i < bits; i++)
    {
        if (br->bits_read >= br->total_bits)
            return v; // safety
        uint8_t cur = br->buf[br->byte_pos];
        int bit = (cur >> br->bit_pos) & 1;
        v |= (bit << i);
        br->bit_pos++;
        if (br->bit_pos == 8)
        {
            br->bit_pos = 0;
            br->byte_pos++;
        }
        br->bits_read++;
    }
    return v;
}
static uint32_t br_get_varint(bit_reader_t *br)
{
    uint32_t v = 0;
    int shift = 0;
    while (1)
    {
        uint32_t byte = br_get_bits(br, 8);
        v |= (byte & 0x7F) << shift;
        if (!(byte & 0x80))
            break;
        shift += 7;
    }
    return v;
}

////////////////////////////////////////////////////////////////////////
// HELPER IMPLEMENTATIONS
////////////////////////////////////////////////////////////////////////
int helper_pack_values(const uint16_t *vals, int count, uint8_t *dst)
{
    int bit_pos_local = 0;
    int total_bits = count * BITS_PER_TOKEN;
    int dst_len = (total_bits + 7) / 8;
    memset(dst, 0, dst_len + 2);
    for (int i = 0; i < count; ++i)
    {
        uint16_t v = vals[i] & 0x3FF;
        int byte_idx = bit_pos_local / 8;
        int bit_off = bit_pos_local % 8;
        dst[byte_idx] |= (v << bit_off) & 0xFF;
        dst[byte_idx + 1] |= (v >> (8 - bit_off)) & 0xFF;
        if (bit_off > 6)
            dst[byte_idx + 2] |= (v >> (16 - bit_off)) & 0xFF;
        bit_pos_local += BITS_PER_TOKEN;
    }
    return dst_len;
}

void helper_flush_rle(FILE *out, int *run_ptr)
{
    int run = *run_ptr;
    while (run > 0)
    {
        int chunk = MIN(run, 32);
        uint8_t control = CONTROL_FLAG | CONTROL_SUBTYPE_RLE | (uint8_t)((chunk - 1) & 0x1F);
        fwrite(&control, 1, 1, out);
        run -= chunk;
    }
    *run_ptr = 0;
}

void helper_unpack_values(const uint8_t *src, int count, uint16_t *dst)
{
    int bit_pos_local = 0;
    for (int i = 0; i < count; ++i)
    {
        int byte_idx = bit_pos_local / 8;
        int bit_off = bit_pos_local % 8;
        uint32_t bits = src[byte_idx] | ((uint32_t)src[byte_idx + 1] << 8) | ((uint32_t)src[byte_idx + 2] << 16);
        bits >>= bit_off;
        dst[i] = (uint16_t)(bits & 0x3FF);
        bit_pos_local += BITS_PER_TOKEN;
    }
}

// Static NumPy header bytes
static const uint8_t numpy_header[HEADER_SIZE] = {
    0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00, 0x76, 0x00, 0x7B, 0x27, 0x64, 0x65, 0x73, 0x63,
    0x72, 0x27, 0x3A, 0x20, 0x27, 0x3C, 0x69, 0x32, 0x27, 0x2C, 0x20, 0x27, 0x66, 0x6F, 0x72, 0x74,
    0x72, 0x61, 0x6E, 0x5F, 0x6F, 0x72, 0x64, 0x65, 0x72, 0x27, 0x3A, 0x20, 0x46, 0x61, 0x6C, 0x73,
    0x65, 0x2C, 0x20, 0x27, 0x73, 0x68, 0x61, 0x70, 0x65, 0x27, 0x3A, 0x20, 0x28, 0x31, 0x32, 0x30,
    0x30, 0x2C, 0x20, 0x38, 0x2C, 0x20, 0x31, 0x36, 0x29, 0x2C, 0x20, 0x7D, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x0A};

//======================================================================

// Thread work item structure
typedef struct
{
    char input_path[MAX_FILENAME];
    char output_path[MAX_FILENAME];
    size_t original_size;
    size_t compressed_size;
    int result;
    uint8_t *mem_data; // optional in-memory buffer (.token.npy contents)
    size_t mem_size;   // size of in-memory buffer
    bool use_memory;   // true if mem_data should be used instead of input_path
} work_item_t;

// Thread pool structure
typedef struct
{
    work_item_t *work_queue;
    int queue_size;
    int queue_head;
    int queue_tail;
    int active_workers;
    int total_files;
    int completed_files;
    mutex_t queue_mutex;
    cond_t work_available;
    cond_t work_complete;
    bool shutdown;
} thread_pool_t;

// Global thread pool
static thread_pool_t g_pool = {0};

// Function prototypes
void print_usage(const char *prog_name);
int compress_file(const char *input_path, const char *output_path);
int decompress_file(const char *input_path, const char *output_path);
void print_version(void);
void pack_frame(const uint8_t *in_bytes, uint8_t *out_packed);
void unpack_frame(const uint8_t *in_packed, uint8_t *out_bytes);
int batch_compress_archives(void);
bool check_file_exists(const char *filename);
int extract_and_process_archive(const char *archive_path); // legacy
int process_tar_gz_in_memory(const char *archive_path);
int compress_buffer(const uint8_t *data, size_t size, const char *output_path);
void add_memory_work_item(const uint8_t *data, size_t size, const char *output_path);
THREAD_RETURN worker_thread(void *arg);
void init_thread_pool(int max_workers);
void cleanup_thread_pool(void);
void add_work_item(const char *input_path, const char *output_path);
void wait_for_completion(void);
void update_progress(void);
size_t get_file_size(const char *filepath);
int get_cpu_count(void);

// Platform-specific threading functions
int mutex_init(mutex_t *mutex);
int mutex_destroy(mutex_t *mutex);
int mutex_lock(mutex_t *mutex);
int mutex_unlock(mutex_t *mutex);
int cond_init(cond_t *cond);
int cond_destroy(cond_t *cond);
int cond_wait(cond_t *cond, mutex_t *mutex);
int cond_signal(cond_t *cond);
int cond_broadcast(cond_t *cond);
int thread_create(thread_t *thread, void *(*start_routine)(void *), void *arg);
int thread_join(thread_t thread);

// Helper function prototypes
int helper_pack_values(const uint16_t *vals, int count, uint8_t *dst);
void helper_flush_rle(FILE *out, int *run_ptr);
void helper_unpack_values(const uint8_t *src, int count, uint16_t *dst);

////////////////////////////////////////////////////////////////////////
// UTILITY FUNCTIONS
////////////////////////////////////////////////////////////////////////

//
// Print usage information
//
void print_usage(const char *prog_name)
{
    fprintf(stderr, "Usage: %s [option] [file]\n", prog_name);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -c <file>  Compress (bitpack + delta encode) the .token.npy file\n");
    fprintf(stderr, "  -d <file>  Decompress the .token.npy.cmp file\n");
    fprintf(stderr, "  -v         Print version\n");
    fprintf(stderr, "  (no args)  Batch process data-0000.tar.gz and data-0001.tar.gz\n");
}

//
// Print version information
//
void print_version(void)
{
    printf("Delta Encoder %s\n", VERSION);
}

//
// Count the number of changed tokens in the mask
//
static int count_changed(const uint8_t *mask)
{
    int count = 0;
    for (int i = 0; i < MASK_SIZE; ++i)
    {
        uint8_t byte = mask[i];
        while (byte)
        {
            count += byte & 1;
            byte >>= 1;
        }
    }
    return count;
}

/*
===============================================================================
Function Name: pack_frame

Description:
    - Packs a 256-byte frame (128 x 16-bit tokens) into 160 bytes by extracting
      the lower 10 bits of each token (little-endian) and bit-packing them densely.

Parameters:
    - in_bytes: Pointer to 256 input bytes (original frame).
    - out_packed: Pointer to 160 output bytes (packed frame).

Return:
    - None (void).
===============================================================================
*/
void pack_frame(const uint8_t *in_bytes, uint8_t *out_packed)
{
    memset(out_packed, 0, PACKED_FRAME_SIZE);
    int bit_pos = 0;
    for (int t = 0; t < TOKENS_PER_FRAME; ++t)
    {
        uint16_t val = in_bytes[2 * t] | (in_bytes[2 * t + 1] << 8);
        val &= 0x3FF; // Mask to 10 bits (assumes values <= 1023)
        int byte_idx = bit_pos / 8;
        int bit_off = bit_pos % 8;
        out_packed[byte_idx] |= (val << bit_off) & 0xFF;
        out_packed[byte_idx + 1] |= (val >> (8 - bit_off)) & 0xFF;
        if (bit_off > 6) // If spans 3 bytes
            out_packed[byte_idx + 2] |= (val >> (16 - bit_off)) & 0xFF;
        bit_pos += BITS_PER_TOKEN;
    }
}

/*
===============================================================================
Function Name: unpack_frame

Description:
    - Unpacks a 160-byte packed frame back into 256 bytes (128 x 16-bit tokens),
      restoring the original little-endian byte format with high 6 bits zeroed.
      Uses a precise bit-extraction loop to avoid over-grabbing bits from neighbors.

Parameters:
    - in_packed: Pointer to 160 input bytes (packed frame).
    - out_bytes: Pointer to 256 output bytes (unpacked frame).

Return:
    - None (void).
===============================================================================
*/
void unpack_frame(const uint8_t *in_packed, uint8_t *out_bytes)
{
    int bit_pos = 0;
    for (int t = 0; t < TOKENS_PER_FRAME; ++t)
    {
        int byte_idx = bit_pos / 8;
        int bit_off = bit_pos % 8;
        uint32_t bits = 0;
        int shift = 0;
        int remaining = BITS_PER_TOKEN;
        int current_byte_idx = byte_idx;
        int current_bit_off = bit_off;
        while (remaining > 0)
        {
            int take = (8 - current_bit_off < remaining) ? 8 - current_bit_off : remaining;
            uint8_t byte = in_packed[current_byte_idx];
            uint32_t part = (byte >> current_bit_off) & ((1u << take) - 1);
            bits |= (part << shift);
            shift += take;
            remaining -= take;
            current_bit_off = 0;
            current_byte_idx++;
        }
        uint16_t val = bits & 0x3FF;     // Ensure only 10 bits
        out_bytes[2 * t] = val & 0xFF;   // Low byte
        out_bytes[2 * t + 1] = val >> 8; // High byte (0-3)
        bit_pos += BITS_PER_TOKEN;
    }
}

/*
===============================================================================
Function Name: compress_file

Description:
    - Compresses the input file: Bit packs each frame (256 -> 160 bytes),
      then delta encodes the packed frames.

Parameters:
    - input_path: The path to the input .token.npy file.
    - output_path: The path to the output compressed file.

Return:
    - 0 on success, or EXIT_FAILURE on failure.
===============================================================================
*/
int compress_file(const char *input_path, const char *output_path)
{
    FILE *in_fp = fopen(input_path, "rb");
    FILE *out_fp = fopen(output_path, "wb");
    if (!in_fp || !out_fp)
        return -1;

    // Determine input size and detect presence of NumPy header
    fseek(in_fp, 0, SEEK_END);
    long input_size = ftell(in_fp);
    fseek(in_fp, 0, SEEK_SET);

    long expected_with_header = HEADER_SIZE + (long)FRAME_SIZE * NUM_FRAMES;
    long expected_no_header = (long)FRAME_SIZE * NUM_FRAMES;
    bool has_header = false;
    if (input_size == expected_with_header)
    {
        has_header = true;
    }
    else if (input_size == expected_no_header)
    {
        has_header = false;
    }
    else
    {
        // Size mismatch – refuse to proceed
        fprintf(stderr, "compress_file: unexpected input size %ld (expected %ld or %ld)\n", input_size, expected_with_header, expected_no_header);
        fclose(in_fp);
        fclose(out_fp);
        return -2;
    }

    if (has_header)
    {
        // Skip NumPy header
        if (fseek(in_fp, HEADER_SIZE, SEEK_SET) != 0)
        {
            fclose(in_fp);
            fclose(out_fp);
            return -2;
        }
    }

    // Load token data into memory (always FRAME_SIZE * NUM_FRAMES bytes)
    size_t data_bytes = FRAME_SIZE * NUM_FRAMES;
    uint8_t *data = malloc(data_bytes);
    if (!data)
    {
        fclose(in_fp);
        fclose(out_fp);
        return -2;
    }
    size_t read_count = fread(data, 1, data_bytes, in_fp);
    if (read_count != data_bytes)
    {
        fprintf(stderr, "compress_file: short read (%zu vs %zu)\n", read_count, data_bytes);
        free(data);
        fclose(in_fp);
        fclose(out_fp);
        return -2;
    }
    fclose(in_fp);

    uint8_t prev_tokens[FRAME_SIZE];

    // New arrays for static token detection
    uint8_t static_mask[MASK_SIZE] = {0};
    uint8_t static_values[FRAME_SIZE] = {0};

    // Write keyframe packed
    uint8_t key_packed[PACKED_FRAME_SIZE];
    pack_frame(data, key_packed);
    fwrite(key_packed, 1, PACKED_FRAME_SIZE, out_fp);

    // Compute static tokens across all frames
    uint8_t *first_frame = data;
    for (int t = 0; t < TOKENS_PER_FRAME; ++t)
    {
        bool is_static = true;
        uint16_t first_val = first_frame[2 * t] | (first_frame[2 * t + 1] << 8);
        first_val &= 0x3FF;
        for (int f = 1; f < NUM_FRAMES; ++f)
        {
            uint8_t *frame = data + f * FRAME_SIZE;
            uint16_t v = frame[2 * t] | (frame[2 * t + 1] << 8);
            if ((v & 0x3FF) != first_val)
            {
                is_static = false;
                break;
            }
        }
        if (is_static)
        {
            static_mask[t / 8] |= (1u << (t % 8));
            static_values[2 * t] = first_frame[2 * t];
            static_values[2 * t + 1] = first_frame[2 * t + 1];
        }
    }

    // Write static mask
    fwrite(static_mask, 1, MASK_SIZE, out_fp);

    // Write static token values (in token order)
    for (int t = 0; t < TOKENS_PER_FRAME; ++t)
    {
        if (static_mask[t / 8] & (1u << (t % 8)))
        {
            fwrite(&static_values[2 * t], 1, 2, out_fp);
        }
    }

    // Prepare for delta frames - previous tokens = first frame
    memcpy(prev_tokens, first_frame, FRAME_SIZE);

    // Build unified opcode bitstream for remaining frames
    bit_writer_t bw;
    bw_init(&bw, 64 * 1024); // initial cap
    if (!bw.buf)
    {
        free(data);
        fclose(out_fp);
        return -3;
    }
    int rle_run = 0;
    for (int f = 1; f < NUM_FRAMES; ++f)
    {
        uint8_t *curr = data + f * FRAME_SIZE;
        int changed_indices[128];
        uint16_t changed_values[128];
        int changed_count = 0;
        uint8_t local_mask[MASK_SIZE];
        memset(local_mask, 0, MASK_SIZE);
        for (int t = 0; t < TOKENS_PER_FRAME; ++t)
        {
            if ((static_mask[t / 8] >> (t % 8)) & 1)
                continue;
            uint16_t pv = (prev_tokens[2 * t] | (prev_tokens[2 * t + 1] << 8)) & 0x3FF;
            uint16_t cv = (curr[2 * t] | (curr[2 * t + 1] << 8)) & 0x3FF;
            if (cv != pv)
            {
                local_mask[t / 8] |= (1u << (t % 8));
                changed_indices[changed_count] = t;
                changed_values[changed_count] = cv;
                changed_count++;
            }
        }
        if (changed_count == 0)
        {
            rle_run++;
            memcpy(prev_tokens, curr, FRAME_SIZE);
            continue;
        }
        // flush RLE
        if (rle_run > 0)
        {
            bw_put_bits(&bw, OPCODE_RLE, 2);
            bw_put_varint(&bw, (uint32_t)(rle_run - 1));
            rle_run = 0;
        }
        if (changed_count <= SPARSE_THRESHOLD)
        {
            bw_put_bits(&bw, OPCODE_SPARSE, 2);
            bw_put_bits(&bw, (uint32_t)(changed_count - 1), 5); // count-1 (supports up to 32)
            for (int i = 0; i < changed_count; ++i)
            {
                bw_put_bits(&bw, (uint32_t)changed_indices[i], 7);
            }
            for (int i = 0; i < changed_count; ++i)
            {
                bw_put_bits(&bw, (uint32_t)changed_values[i], 10);
            }
        }
        else
        {
            bw_put_bits(&bw, OPCODE_DENSE, 2);
            for (int i = 0; i < MASK_SIZE; ++i)
                bw_put_bits(&bw, local_mask[i], 8);
            for (int i = 0; i < changed_count; ++i)
            {
                bw_put_bits(&bw, (uint32_t)changed_values[i], 10);
            }
        }
        memcpy(prev_tokens, curr, FRAME_SIZE);
    }
    if (rle_run > 0)
    {
        bw_put_bits(&bw, OPCODE_RLE, 2);
        bw_put_varint(&bw, (uint32_t)(rle_run - 1));
    }
    uint32_t bit_count = bw_finish(&bw);
    fwrite(&bit_count, 1, 4, out_fp); // store number of valid bits
    size_t stream_bytes = (bit_count + 7) / 8;
    fwrite(bw.buf, 1, stream_bytes, out_fp);
    bw_free(&bw);

    free(data);
    fclose(out_fp);
    return 0;
}

/*
===============================================================================
Function Name: compress_buffer

Description:
    - Compresses an in-memory buffer containing exactly the raw token bytes or
      a NumPy .npy header followed by the raw token bytes (same sizes validated
      in compress_file). Output format identical to compress_file.

Parameters:
    - data: Pointer to input bytes (may include numpy header)
    - size: Size of the buffer in bytes
    - output_path: Destination file path

Return:
    - 0 on success, non‑zero on failure
===============================================================================
*/
int compress_buffer(const uint8_t *data, size_t size, const char *output_path)
{
    if (!data || !output_path)
        return -1;

    FILE *out_fp = fopen(output_path, "wb");
    if (!out_fp)
        return -1;

    size_t expected_with_header = HEADER_SIZE + (size_t)FRAME_SIZE * NUM_FRAMES;
    size_t expected_no_header = (size_t)FRAME_SIZE * NUM_FRAMES;
    bool has_header = false;
    if (size == expected_with_header)
        has_header = true;
    else if (size == expected_no_header)
        has_header = false;
    else
    {
        fprintf(stderr, "compress_buffer: unexpected input size %zu (expected %zu or %zu)\n", size, expected_with_header, expected_no_header);
        fclose(out_fp);
        return -2;
    }

    const uint8_t *frames = data + (has_header ? HEADER_SIZE : 0);

    uint8_t prev_tokens[FRAME_SIZE];
    uint8_t static_mask[MASK_SIZE] = {0};
    uint8_t static_values[FRAME_SIZE] = {0};

    // Keyframe
    uint8_t key_packed[PACKED_FRAME_SIZE];
    pack_frame(frames, key_packed);
    fwrite(key_packed, 1, PACKED_FRAME_SIZE, out_fp);

    // Static token detection
    const uint8_t *first_frame = frames;
    for (int t = 0; t < TOKENS_PER_FRAME; ++t)
    {
        bool is_static = true;
        uint16_t first_val = first_frame[2 * t] | (first_frame[2 * t + 1] << 8);
        first_val &= 0x3FF;
        for (int f = 1; f < NUM_FRAMES; ++f)
        {
            const uint8_t *frame = frames + f * FRAME_SIZE;
            uint16_t v = frame[2 * t] | (frame[2 * t + 1] << 8);
            if ((v & 0x3FF) != first_val)
            {
                is_static = false;
                break;
            }
        }
        if (is_static)
        {
            static_mask[t / 8] |= (1u << (t % 8));
            static_values[2 * t] = first_frame[2 * t];
            static_values[2 * t + 1] = first_frame[2 * t + 1];
        }
    }

    fwrite(static_mask, 1, MASK_SIZE, out_fp);
    for (int t = 0; t < TOKENS_PER_FRAME; ++t)
        if (static_mask[t / 8] & (1u << (t % 8)))
            fwrite(&static_values[2 * t], 1, 2, out_fp);

    memcpy(prev_tokens, first_frame, FRAME_SIZE);

    bit_writer_t bw;
    bw_init(&bw, 64 * 1024);
    if (!bw.buf)
    {
        fclose(out_fp);
        return -3;
    }
    int rle_run = 0;
    for (int f = 1; f < NUM_FRAMES; ++f)
    {
        const uint8_t *curr = frames + f * FRAME_SIZE;
        int changed_indices[128];
        uint16_t changed_values[128];
        int changed_count = 0;
        uint8_t local_mask[MASK_SIZE];
        memset(local_mask, 0, MASK_SIZE);
        for (int t = 0; t < TOKENS_PER_FRAME; ++t)
        {
            if ((static_mask[t / 8] >> (t % 8)) & 1)
                continue;
            uint16_t pv = (prev_tokens[2 * t] | (prev_tokens[2 * t + 1] << 8)) & 0x3FF;
            uint16_t cv = (curr[2 * t] | (curr[2 * t + 1] << 8)) & 0x3FF;
            if (cv != pv)
            {
                local_mask[t / 8] |= (1u << (t % 8));
                changed_indices[changed_count] = t;
                changed_values[changed_count] = cv;
                changed_count++;
            }
        }
        if (changed_count == 0)
        {
            rle_run++;
            memcpy(prev_tokens, curr, FRAME_SIZE);
            continue;
        }
        if (rle_run > 0)
        {
            bw_put_bits(&bw, OPCODE_RLE, 2);
            bw_put_varint(&bw, (uint32_t)(rle_run - 1));
            rle_run = 0;
        }
        if (changed_count <= SPARSE_THRESHOLD)
        {
            bw_put_bits(&bw, OPCODE_SPARSE, 2);
            bw_put_bits(&bw, (uint32_t)(changed_count - 1), 5);
            for (int i = 0; i < changed_count; ++i)
                bw_put_bits(&bw, (uint32_t)changed_indices[i], 7);
            for (int i = 0; i < changed_count; ++i)
                bw_put_bits(&bw, (uint32_t)changed_values[i], 10);
        }
        else
        {
            bw_put_bits(&bw, OPCODE_DENSE, 2);
            for (int i = 0; i < MASK_SIZE; ++i)
                bw_put_bits(&bw, local_mask[i], 8);
            for (int i = 0; i < changed_count; ++i)
                bw_put_bits(&bw, (uint32_t)changed_values[i], 10);
        }
        memcpy(prev_tokens, curr, FRAME_SIZE);
    }
    if (rle_run > 0)
    {
        bw_put_bits(&bw, OPCODE_RLE, 2);
        bw_put_varint(&bw, (uint32_t)(rle_run - 1));
    }
    uint32_t bit_count = bw_finish(&bw);
    fwrite(&bit_count, 1, 4, out_fp);
    size_t stream_bytes = (bit_count + 7) / 8;
    fwrite(bw.buf, 1, stream_bytes, out_fp);
    bw_free(&bw);
    fclose(out_fp);
    return 0;
}

/*
===============================================================================
Function Name: decompress_file

Description:
    - Decompresses the input file: Delta decodes to packed frames,
      then unpacks each to original 256 bytes.

Parameters:
    - input_path: The path to the input compressed file.
    - output_path: The path to the output .token.npy file.

Return:
    - 0 on success, or EXIT_FAILURE on failure.
===============================================================================
*/
int decompress_file(const char *input_path, const char *output_path)
{
    FILE *in_fp = fopen(input_path, "rb");
    FILE *out_fp = fopen(output_path, "wb");
    if (!in_fp || !out_fp)
        return -1;

    // Load entire compressed file to memory for easy random access
    fseek(in_fp, 0, SEEK_END);
    long file_size = ftell(in_fp);
    fseek(in_fp, 0, SEEK_SET);

    uint8_t *comp_data = malloc(file_size);
    if (!comp_data)
        return -2;
    fread(comp_data, 1, file_size, in_fp);
    fclose(in_fp);

    uint8_t *out_data = malloc(FRAME_SIZE * NUM_FRAMES);
    if (!out_data)
    {
        free(comp_data);
        return -3;
    }

    uint8_t prev_frame[FRAME_SIZE];
    uint8_t curr_frame[FRAME_SIZE];
    // mask removed (unused in new bitstream format)
    uint8_t static_mask[MASK_SIZE] = {0};

    size_t comp_pos = 0;

    // Read keyframe packed and unpack it (use your unpack_frame)
    unpack_frame(comp_data + comp_pos, out_data);
    comp_pos += PACKED_FRAME_SIZE;

    memcpy(prev_frame, out_data, FRAME_SIZE);

    // Read static mask
    memcpy(static_mask, comp_data + comp_pos, MASK_SIZE);
    comp_pos += MASK_SIZE;

    // Read static values
    for (int t = 0; t < TOKENS_PER_FRAME; ++t)
    {
        if (static_mask[t / 8] & (1u << (t % 8)))
        {
            out_data[2 * t] = comp_data[comp_pos++];
            out_data[2 * t + 1] = comp_data[comp_pos++];
        }
    }

    memcpy(prev_frame, out_data, FRAME_SIZE);

    // Read bitstream length (bits)
    if (comp_pos + 4 > (size_t)file_size)
    {
        free(out_data);
        free(comp_data);
        fclose(out_fp);
        return -4;
    }
    uint32_t bit_count = *(uint32_t *)(comp_data + comp_pos);
    comp_pos += 4;
    size_t stream_bytes = (bit_count + 7) / 8;
    if (comp_pos + stream_bytes > (size_t)file_size)
    {
        free(out_data);
        free(comp_data);
        fclose(out_fp);
        return -5;
    }
    bit_reader_t br;
    br_init(&br, comp_data + comp_pos, stream_bytes, bit_count);
    int frames_decoded = 1; // already have frame 0
    while (frames_decoded < NUM_FRAMES && br.bits_read < br.total_bits)
    {
        uint32_t op = br_get_bits(&br, 2);
        if (op == OPCODE_RLE)
        {
            uint32_t run = br_get_varint(&br) + 1; // stored as run-1
            for (uint32_t r = 0; r < run && frames_decoded < NUM_FRAMES; ++r)
            {
                memcpy(out_data + frames_decoded * FRAME_SIZE, prev_frame, FRAME_SIZE);
                frames_decoded++;
            }
        }
        else if (op == OPCODE_SPARSE)
        {
            uint32_t count = br_get_bits(&br, 5) + 1;
            if (count > 32)
                break;
            int indices[32];
            for (uint32_t i = 0; i < count; ++i)
                indices[i] = (int)br_get_bits(&br, 7);
            memcpy(curr_frame, prev_frame, FRAME_SIZE);
            for (uint32_t i = 0; i < count; ++i)
            {
                int t = indices[i];
                if ((static_mask[t / 8] >> (t % 8)) & 1)
                {
                    (void)br_get_bits(&br, 10);
                    continue;
                }
                uint16_t v = (uint16_t)br_get_bits(&br, 10);
                curr_frame[2 * t] = v & 0xFF;
                curr_frame[2 * t + 1] = (v >> 8) & 0xFF;
            }
            memcpy(prev_frame, curr_frame, FRAME_SIZE);
            memcpy(out_data + frames_decoded * FRAME_SIZE, curr_frame, FRAME_SIZE);
            frames_decoded++;
        }
        else if (op == OPCODE_DENSE)
        {
            memcpy(curr_frame, prev_frame, FRAME_SIZE);
            uint8_t local_mask[MASK_SIZE];
            for (int i = 0; i < MASK_SIZE; ++i)
                local_mask[i] = (uint8_t)br_get_bits(&br, 8);
            for (int t = 0; t < TOKENS_PER_FRAME; ++t)
            {
                if ((static_mask[t / 8] >> (t % 8)) & 1)
                    continue;
                if (local_mask[t / 8] & (1u << (t % 8)))
                {
                    uint16_t v = (uint16_t)br_get_bits(&br, 10);
                    curr_frame[2 * t] = v & 0xFF;
                    curr_frame[2 * t + 1] = (v >> 8) & 0xFF;
                }
            }
            memcpy(prev_frame, curr_frame, FRAME_SIZE);
            memcpy(out_data + frames_decoded * FRAME_SIZE, curr_frame, FRAME_SIZE);
            frames_decoded++;
        }
        else
        { // reserved / unknown
            break;
        }
    }

    // Prepend NumPy header so output is a valid .npy file
    fwrite(numpy_header, 1, HEADER_SIZE, out_fp);
    fwrite(out_data, 1, FRAME_SIZE * NUM_FRAMES, out_fp);
    free(out_data);
    free(comp_data);
    fclose(out_fp);
    return 0;
}

////////////////////////////////////////////////////////////////////////
// PLATFORM-SPECIFIC THREADING FUNCTIONS
////////////////////////////////////////////////////////////////////////

//
// Get CPU count
//
int get_cpu_count(void)
{
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return (int)sysinfo.dwNumberOfProcessors;
#else
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

//
// Mutex functions
//
int mutex_init(mutex_t *mutex)
{
#ifdef _WIN32
    InitializeCriticalSection(mutex);
    return 0;
#else
    return pthread_mutex_init(mutex, NULL);
#endif
}

int mutex_destroy(mutex_t *mutex)
{
#ifdef _WIN32
    DeleteCriticalSection(mutex);
    return 0;
#else
    return pthread_mutex_destroy(mutex);
#endif
}

int mutex_lock(mutex_t *mutex)
{
#ifdef _WIN32
    EnterCriticalSection(mutex);
    return 0;
#else
    return pthread_mutex_lock(mutex);
#endif
}

int mutex_unlock(mutex_t *mutex)
{
#ifdef _WIN32
    LeaveCriticalSection(mutex);
    return 0;
#else
    return pthread_mutex_unlock(mutex);
#endif
}

//
// Condition variable functions
//
int cond_init(cond_t *cond)
{
#ifdef _WIN32
    InitializeConditionVariable(cond);
    return 0;
#else
    return pthread_cond_init(cond, NULL);
#endif
}

int cond_destroy(cond_t *cond)
{
#ifdef _WIN32
    // Windows condition variables don't need explicit cleanup
    (void)cond;
    return 0;
#else
    return pthread_cond_destroy(cond);
#endif
}

int cond_wait(cond_t *cond, mutex_t *mutex)
{
#ifdef _WIN32
    return SleepConditionVariableCS(cond, mutex, INFINITE) ? 0 : -1;
#else
    return pthread_cond_wait(cond, mutex);
#endif
}

int cond_signal(cond_t *cond)
{
#ifdef _WIN32
    WakeConditionVariable(cond);
    return 0;
#else
    return pthread_cond_signal(cond);
#endif
}

int cond_broadcast(cond_t *cond)
{
#ifdef _WIN32
    WakeAllConditionVariable(cond);
    return 0;
#else
    return pthread_cond_broadcast(cond);
#endif
}

//
// Thread functions
//
int thread_create(thread_t *thread, void *(*start_routine)(void *), void *arg)
{
#ifdef _WIN32
    *thread = (HANDLE)_beginthreadex(NULL, 0, (unsigned int(__stdcall *)(void *))start_routine, arg, 0, NULL);
    return (*thread != 0) ? 0 : -1;
#else
    return pthread_create(thread, NULL, start_routine, arg);
#endif
}

int thread_join(thread_t thread)
{
#ifdef _WIN32
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return 0;
#else
    return pthread_join(thread, NULL);
#endif
}

////////////////////////////////////////////////////////////////////////
// BATCH PROCESSING FUNCTIONS
////////////////////////////////////////////////////////////////////////

//
// Check if a file exists
//
bool check_file_exists(const char *filename)
{
    struct stat st;
    return (stat(filename, &st) == 0);
}

//
// Get file size
//
size_t get_file_size(const char *filepath)
{
    struct stat st;
    if (stat(filepath, &st) == 0)
    {
        return st.st_size;
    }
    return 0;
}

//
// Initialize thread pool
//
void init_thread_pool(int max_workers)
{
    g_pool.work_queue = malloc(sizeof(work_item_t) * 10000); // Max 10k files
    g_pool.queue_size = 10000;
    g_pool.queue_head = 0;
    g_pool.queue_tail = 0;
    g_pool.active_workers = max_workers;
    g_pool.total_files = 0;
    g_pool.completed_files = 0;
    g_pool.shutdown = false;

    mutex_init(&g_pool.queue_mutex);
    cond_init(&g_pool.work_available);
    cond_init(&g_pool.work_complete);
}

//
// Cleanup thread pool
//
void cleanup_thread_pool(void)
{
    if (g_pool.work_queue)
    {
        free(g_pool.work_queue);
        g_pool.work_queue = NULL;
    }

    mutex_destroy(&g_pool.queue_mutex);
    cond_destroy(&g_pool.work_available);
    cond_destroy(&g_pool.work_complete);
}

//
// Worker thread function
//
THREAD_RETURN worker_thread(void *arg)
{
    (void)arg; // Unused parameter

    while (true)
    {
        mutex_lock(&g_pool.queue_mutex);

        // Wait for work or shutdown signal
        while (g_pool.queue_head == g_pool.queue_tail && !g_pool.shutdown)
        {
            cond_wait(&g_pool.work_available, &g_pool.queue_mutex);
        }

        if (g_pool.shutdown)
        {
            mutex_unlock(&g_pool.queue_mutex);
            break;
        }

        // Get work item
        work_item_t work = g_pool.work_queue[g_pool.queue_head];
        g_pool.queue_head = (g_pool.queue_head + 1) % g_pool.queue_size;

        mutex_unlock(&g_pool.queue_mutex);

        // Do the work
        if (work.use_memory)
        {
            work.result = compress_buffer(work.mem_data, work.mem_size, work.output_path);
            if (work.mem_data)
                free(work.mem_data);
            work.original_size = work.mem_size;
            work.compressed_size = get_file_size(work.output_path);
        }
        else
        {
            work.original_size = get_file_size(work.input_path);
            work.result = compress_file(work.input_path, work.output_path);
            work.compressed_size = get_file_size(work.output_path);
        }

        // Update progress
        mutex_lock(&g_pool.queue_mutex);
        g_pool.completed_files++;
        cond_signal(&g_pool.work_complete);
        mutex_unlock(&g_pool.queue_mutex);
    }

#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

//
// Add work item to queue
//
void add_work_item(const char *input_path, const char *output_path)
{
    mutex_lock(&g_pool.queue_mutex);

    work_item_t *work = &g_pool.work_queue[g_pool.queue_tail];
    strncpy(work->input_path, input_path, MAX_FILENAME - 1);
    strncpy(work->output_path, output_path, MAX_FILENAME - 1);
    work->input_path[MAX_FILENAME - 1] = '\0';
    work->output_path[MAX_FILENAME - 1] = '\0';
    work->use_memory = false;
    work->mem_data = NULL;
    work->mem_size = 0;

    g_pool.queue_tail = (g_pool.queue_tail + 1) % g_pool.queue_size;
    g_pool.total_files++;

    cond_signal(&g_pool.work_available);
    mutex_unlock(&g_pool.queue_mutex);
}

void add_memory_work_item(const uint8_t *data, size_t size, const char *output_path)
{
    mutex_lock(&g_pool.queue_mutex);
    work_item_t *work = &g_pool.work_queue[g_pool.queue_tail];
    work->input_path[0] = '\0';
    strncpy(work->output_path, output_path, MAX_FILENAME - 1);
    work->output_path[MAX_FILENAME - 1] = '\0';
    work->original_size = size;
    work->compressed_size = 0;
    work->result = 0;
    work->use_memory = true;
    work->mem_size = size;
    work->mem_data = (uint8_t *)malloc(size);
    if (work->mem_data)
        memcpy(work->mem_data, data, size);
    g_pool.queue_tail = (g_pool.queue_tail + 1) % g_pool.queue_size;
    g_pool.total_files++;
    cond_signal(&g_pool.work_available);
    mutex_unlock(&g_pool.queue_mutex);
}

//
// Update progress bar
//
void update_progress(void)
{
    static time_t last_update = 0;
    time_t now = time(NULL);

    if (now - last_update >= 1)
    { // Update every second
        mutex_lock(&g_pool.queue_mutex);
        int completed = g_pool.completed_files;
        int total = g_pool.total_files;
        mutex_unlock(&g_pool.queue_mutex);

        if (total > 0)
        {
            float percent = (float)completed / total * 100.0f;
            int bar_width = 50;
            int filled = (int)(percent / 100.0f * bar_width);

            printf("\rProgress: [");
            for (int i = 0; i < bar_width; i++)
            {
                if (i < filled)
                    printf("=");
                else if (i == filled)
                    printf(">");
                else
                    printf(" ");
            }
            printf("] %d/%d (%.1f%%)", completed, total, percent);
            fflush(stdout);
        }

        last_update = now;
    }
}

//
// Wait for all work to complete
//
void wait_for_completion(void)
{
    mutex_lock(&g_pool.queue_mutex);

    while (g_pool.completed_files < g_pool.total_files)
    {
        update_progress();
        cond_wait(&g_pool.work_complete, &g_pool.queue_mutex);
    }

    mutex_unlock(&g_pool.queue_mutex);

    // Final progress update
    printf("\rProgress: [");
    for (int i = 0; i < 50; i++)
        printf("=");
    printf("] %d/%d (100.0%%)\n", g_pool.total_files, g_pool.total_files);
}

//
// Extract and process archive
//
int extract_and_process_archive(const char *archive_path)
{
    // Legacy path disabled; use in-memory streaming instead
    (void)archive_path;
    return EXIT_FAILURE;
}

// TAR header structure
typedef struct
{
    char name[100];
    char mode[8];
    char uid[8];
    char gid[8];
    char size[12];
    char mtime[12];
    char chksum[8];
    char typeflag;
    char linkname[100];
    char magic[6];
    char version[2];
    char uname[32];
    char gname[32];
    char devmajor[8];
    char devminor[8];
    char prefix[155];
    char pad[12];
} tar_header_t;
static size_t tar_parse_octal(const char *s, size_t n)
{
    size_t v = 0;
    for (size_t i = 0; i < n && s[i]; ++i)
    {
        if (s[i] < '0' || s[i] > '7')
            break;
        v = (v << 3) + (size_t)(s[i] - '0');
    }
    return v;
}
static char *tar_base(char *p)
{
    char *a = strrchr(p, '/');
    char *b = strrchr(p, '\\');
    char *m = a > b ? a : b;
    return m ? m + 1 : p;
}
static void derive_output_dir(const char *archive, char *outdir, size_t cap)
{
    const char *b = tar_base((char *)archive);
    size_t len = strlen(b);
    if (len > 7 && strcmp(b + len - 7, ".tar.gz") == 0)
        len -= 7;
    if (len >= cap)
        len = cap - 1;
    memcpy(outdir, b, len);
    outdir[len] = '\0';
}
static int make_dir(const char *path)
{
#ifdef _WIN32
    if (_mkdir(path) == 0 || errno == EEXIST)
        return 0;
#else
    if (mkdir(path, 0755) == 0 || errno == EEXIST)
        return 0;
#endif
    return -1;
}

int process_tar_gz_in_memory(const char *archive_path)
{
    gzFile gzf = gzopen(archive_path, "rb");
    if (!gzf)
    {
        fprintf(stderr, "Failed to open %s\n", archive_path);
        return EXIT_FAILURE;
    }
    char outdir[256];
    derive_output_dir(archive_path, outdir, sizeof(outdir));
    make_dir(outdir);
    const size_t BLK = 512;
    uint8_t hdr[512];
    while (1)
    {
        int r = gzread(gzf, hdr, BLK);
        if (r == 0)
            break;
        if (r != (int)BLK)
        {
            fprintf(stderr, "Short read header %s\n", archive_path);
            break;
        }
        bool zero = true;
        for (int i = 0; i < 512; i++)
        {
            if (hdr[i] != 0)
            {
                zero = false;
                break;
            }
        }
        if (zero)
        {
            gzread(gzf, hdr, BLK);
            break;
        }
        tar_header_t *th = (tar_header_t *)hdr;
        size_t fsize = tar_parse_octal(th->size, sizeof(th->size));
        char fname[256] = {0};
        if (th->prefix[0])
            snprintf(fname, sizeof(fname), "%s/%s", th->prefix, th->name);
        else
            snprintf(fname, sizeof(fname), "%s", th->name);
        size_t aligned = ((fsize + BLK - 1) / BLK) * BLK;
        if (th->typeflag == '0' || th->typeflag == '\0')
        {
            if (strstr(fname, ".token.npy"))
            {
                if (fsize != (size_t)FRAME_SIZE * NUM_FRAMES && fsize != HEADER_SIZE + (size_t)FRAME_SIZE * NUM_FRAMES)
                {
                    size_t to_skip = aligned;
                    while (to_skip > 0)
                    {
                        size_t chunk = to_skip > BLK ? BLK : to_skip;
                        uint8_t tmp[512];
                        int rr = gzread(gzf, tmp, (unsigned)chunk);
                        if (rr <= 0)
                            break;
                        to_skip -= rr;
                    }
                }
                else
                {
                    uint8_t *buf = (uint8_t *)malloc(fsize);
                    if (!buf)
                    {
                        gzclose(gzf);
                        return EXIT_FAILURE;
                    }
                    size_t left = fsize, off = 0;
                    while (left > 0)
                    {
                        int chunk = (int)MIN(left, (size_t)65536);
                        int rr = gzread(gzf, buf + off, chunk);
                        if (rr <= 0)
                        {
                            free(buf);
                            gzclose(gzf);
                            return EXIT_FAILURE;
                        }
                        off += rr;
                        left -= rr;
                    }
                    size_t pad = aligned - fsize;
                    if (pad)
                    {
                        uint8_t tmp[512];
                        size_t to_skip = pad;
                        while (to_skip > 0)
                        {
                            size_t chunk = to_skip > BLK ? BLK : to_skip;
                            int rr = gzread(gzf, tmp, (unsigned)chunk);
                            if (rr <= 0)
                                break;
                            to_skip -= rr;
                        }
                    }
                    char outpath[MAX_FILENAME];
                    snprintf(outpath, sizeof(outpath), "%s/%s.cmp", outdir, tar_base(fname));
                    add_memory_work_item(buf, fsize, outpath);
                    free(buf);
                }
            }
            else
            {
                size_t to_skip = aligned;
                while (to_skip > 0)
                {
                    size_t chunk = to_skip > BLK ? BLK : to_skip;
                    uint8_t tmp[512];
                    int rr = gzread(gzf, tmp, (unsigned)chunk);
                    if (rr <= 0)
                        break;
                    to_skip -= rr;
                }
            }
        }
        else
        {
            size_t to_skip = aligned;
            while (to_skip > 0)
            {
                size_t chunk = to_skip > BLK ? BLK : to_skip;
                uint8_t tmp[512];
                int rr = gzread(gzf, tmp, (unsigned)chunk);
                if (rr <= 0)
                    break;
                to_skip -= rr;
            }
        }
    }
    gzclose(gzf);
    return EXIT_SUCCESS;
}

int batch_compress_archives(void)
{
    printf("CommaVQ Batch Processor %s\n", VERSION);
    printf("Streaming archives in-memory...\n");
    if (!check_file_exists("data-0000.tar.gz") || !check_file_exists("data-0001.tar.gz"))
    {
        fprintf(stderr, "Archives missing.\n");
        return EXIT_FAILURE;
    }
    int threads = get_cpu_count();
    init_thread_pool(threads);
    thread_t *workers = malloc(sizeof(thread_t) * threads);
    if (!workers)
    {
        fprintf(stderr, "Thread alloc failed\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < threads; i++)
        if (thread_create(&workers[i], worker_thread, NULL) != 0)
        {
            fprintf(stderr, "Thread create failed %d\n", i);
            free(workers);
            return EXIT_FAILURE;
        }
    process_tar_gz_in_memory("data-0000.tar.gz");
    process_tar_gz_in_memory("data-0001.tar.gz");
    wait_for_completion();
    mutex_lock(&g_pool.queue_mutex);
    g_pool.shutdown = true;
    cond_broadcast(&g_pool.work_available);
    mutex_unlock(&g_pool.queue_mutex);
    for (int i = 0; i < threads; i++)
        thread_join(workers[i]);
    free(workers);
    size_t total_original = (size_t)g_pool.total_files * 1200 * 128 * 10 / 8;
    size_t total_compressed = 0;
    for (int i = 0; i < g_pool.total_files; ++i)
        total_compressed += g_pool.work_queue[i].compressed_size;
    float ratio = total_compressed ? (float)total_original / total_compressed : 0.f;
    printf("Processed %d files. Compressed %zu bytes → %zu bytes (%.2fx)\n", g_pool.total_files, total_original, total_compressed, ratio);
    cleanup_thread_pool();
    return EXIT_SUCCESS;
}
////////////////////////////////////////////////////////////////////////
// MAIN ENTRY POINT
////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    // Handle batch mode (no arguments)
    if (argc == 1)
    {
        return batch_compress_archives();
    }

    if (argc < 3)
    {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const char *flag = argv[1];

    if (strcmp(flag, "-v") == 0)
    {
        print_version();
        return EXIT_SUCCESS;
    }
    else if (strcmp(flag, "-c") == 0)
    {
        const char *input_path = argv[2];
        char output_path[256];
        snprintf(output_path, sizeof(output_path), "%s.cmp", input_path);
        return compress_file(input_path, output_path);
    }
    else if (strcmp(flag, "-d") == 0)
    {
        const char *input_path = argv[2];
        char output_path[256];
        snprintf(output_path, sizeof(output_path), "%s.dec", input_path);
        return decompress_file(input_path, output_path);
    }
    else
    {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
}