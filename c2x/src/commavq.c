// commavq.c

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <process.h>
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

#define VERSION "v0.1-bitpack-fixed"
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
#define MASK_SIZE (PACKED_FRAME_SIZE / 8) // 20

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
int extract_and_process_archive(const char *archive_path);
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
    // Open input file
    FILE *in_fp = fopen(input_path, "rb");
    if (!in_fp)
    {
        perror("Failed to open input file");
        return EXIT_FAILURE;
    }

    // Check file size
    fseek(in_fp, 0, SEEK_END);
    long file_size = ftell(in_fp);
    if (file_size != TOTAL_FILE_SIZE)
    {
        fprintf(stderr, "Invalid file size: expected %d, got %ld\n", TOTAL_FILE_SIZE, file_size);
        fclose(in_fp);
        return EXIT_FAILURE;
    }
    rewind(in_fp);

    // Skip header
    fseek(in_fp, HEADER_SIZE, SEEK_SET);

    // Read all data into memory
    uint8_t *data = malloc(TOTAL_DATA_SIZE);
    if (!data)
    {
        perror("Failed to allocate memory");
        fclose(in_fp);
        return EXIT_FAILURE;
    }
    if (fread(data, 1, TOTAL_DATA_SIZE, in_fp) != TOTAL_DATA_SIZE)
    {
        perror("Failed to read data");
        free(data);
        fclose(in_fp);
        return EXIT_FAILURE;
    }
    fclose(in_fp);

    // Bit pack all frames
    uint8_t *packed_data = malloc(PACKED_TOTAL_DATA_SIZE);
    if (!packed_data)
    {
        perror("Failed to allocate packed memory");
        free(data);
        return EXIT_FAILURE;
    }
    for (int f = 0; f < NUM_FRAMES; ++f)
    {
        pack_frame(data + f * FRAME_SIZE, packed_data + f * PACKED_FRAME_SIZE);
    }
    free(data); // No longer needed

    // Open output file
    FILE *out_fp = fopen(output_path, "wb");
    if (!out_fp)
    {
        perror("Failed to open output file");
        free(packed_data);
        return EXIT_FAILURE;
    }

    // Write keyframe (first packed frame, full 160 bytes)
    if (fwrite(packed_data, 1, PACKED_FRAME_SIZE, out_fp) != PACKED_FRAME_SIZE)
    {
        perror("Failed to write keyframe");
        free(packed_data);
        fclose(out_fp);
        return EXIT_FAILURE;
    }

    // Process delta frames on packed data
    uint8_t prev_frame[PACKED_FRAME_SIZE];
    memcpy(prev_frame, packed_data, PACKED_FRAME_SIZE);

    for (int f = 1; f < NUM_FRAMES; ++f)
    {
        uint8_t *curr_frame = packed_data + f * PACKED_FRAME_SIZE;
        uint8_t mask[MASK_SIZE] = {0}; // Bitmask for changes (20 bytes)
        uint8_t changed_bytes[PACKED_FRAME_SIZE];
        int changed_count = 0;

        for (int i = 0; i < PACKED_FRAME_SIZE; ++i)
        {
            if (curr_frame[i] != prev_frame[i])
            {
                mask[i / 8] |= (1u << (i % 8));
                changed_bytes[changed_count++] = curr_frame[i];
            }
        }

        // Write mask (20 bytes)
        if (fwrite(mask, 1, MASK_SIZE, out_fp) != MASK_SIZE)
        {
            perror("Failed to write mask");
            free(packed_data);
            fclose(out_fp);
            return EXIT_FAILURE;
        }

        // Write changed bytes (variable length)
        if (fwrite(changed_bytes, 1, changed_count, out_fp) != (size_t)changed_count)
        {
            perror("Failed to write changed bytes");
            free(packed_data);
            fclose(out_fp);
            return EXIT_FAILURE;
        }

        // Update previous frame for next iteration
        memcpy(prev_frame, curr_frame, PACKED_FRAME_SIZE);
    }

    free(packed_data);
    fclose(out_fp);
    printf("Compressed file written to %s\n", output_path);
    return EXIT_SUCCESS;
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
    // Open input file
    FILE *in_fp = fopen(input_path, "rb");
    if (!in_fp)
    {
        perror("Failed to open input file");
        return EXIT_FAILURE;
    }

    // Read entire compressed data into memory
    fseek(in_fp, 0, SEEK_END);
    long comp_size = ftell(in_fp);
    rewind(in_fp);
    uint8_t *comp_data = malloc(comp_size);
    if (!comp_data)
    {
        perror("Failed to allocate memory");
        fclose(in_fp);
        return EXIT_FAILURE;
    }
    if (fread(comp_data, 1, comp_size, in_fp) != (size_t)comp_size)
    {
        perror("Failed to read compressed data");
        free(comp_data);
        fclose(in_fp);
        return EXIT_FAILURE;
    }
    fclose(in_fp);

    // Allocate packed output buffer
    uint8_t *packed_data = malloc(PACKED_TOTAL_DATA_SIZE);
    if (!packed_data)
    {
        perror("Failed to allocate packed memory");
        free(comp_data);
        return EXIT_FAILURE;
    }

    // Read keyframe
    memcpy(packed_data, comp_data, PACKED_FRAME_SIZE);
    size_t comp_pos = PACKED_FRAME_SIZE;

    // Reconstruct delta frames
    uint8_t prev_frame[PACKED_FRAME_SIZE];
    memcpy(prev_frame, packed_data, PACKED_FRAME_SIZE);

    for (int f = 1; f < NUM_FRAMES; ++f)
    {
        if (comp_pos + MASK_SIZE > (size_t)comp_size)
        {
            fprintf(stderr, "Unexpected end of compressed data\n");
            free(comp_data);
            free(packed_data);
            return EXIT_FAILURE;
        }

        uint8_t mask[MASK_SIZE];
        memcpy(mask, comp_data + comp_pos, MASK_SIZE);
        comp_pos += MASK_SIZE;

        uint8_t *curr_frame = packed_data + f * PACKED_FRAME_SIZE;
        memcpy(curr_frame, prev_frame, PACKED_FRAME_SIZE); // Start with previous frame

        for (int i = 0; i < PACKED_FRAME_SIZE; ++i)
        {
            if (mask[i / 8] & (1u << (i % 8)))
            {
                if (comp_pos >= (size_t)comp_size)
                {
                    fprintf(stderr, "Unexpected end of changed bytes\n");
                    free(comp_data);
                    free(packed_data);
                    return EXIT_FAILURE;
                }
                curr_frame[i] = comp_data[comp_pos++];
            }
        }

        // Update previous frame
        memcpy(prev_frame, curr_frame, PACKED_FRAME_SIZE);
    }

    free(comp_data);

    // Unpack to original data
    uint8_t *out_data = malloc(TOTAL_DATA_SIZE);
    if (!out_data)
    {
        perror("Failed to allocate output memory");
        free(packed_data);
        return EXIT_FAILURE;
    }
    for (int f = 0; f < NUM_FRAMES; ++f)
    {
        unpack_frame(packed_data + f * PACKED_FRAME_SIZE, out_data + f * FRAME_SIZE);
    }
    free(packed_data);

    // Open output file
    FILE *out_fp = fopen(output_path, "wb");
    if (!out_fp)
    {
        perror("Failed to open output file");
        free(out_data);
        return EXIT_FAILURE;
    }

    // Write static NumPy header
    if (fwrite(numpy_header, 1, HEADER_SIZE, out_fp) != HEADER_SIZE)
    {
        perror("Failed to write header");
        free(out_data);
        fclose(out_fp);
        return EXIT_FAILURE;
    }

    // Write reconstructed data
    if (fwrite(out_data, 1, TOTAL_DATA_SIZE, out_fp) != TOTAL_DATA_SIZE)
    {
        perror("Failed to write data");
        free(out_data);
        fclose(out_fp);
        return EXIT_FAILURE;
    }

    free(out_data);
    fclose(out_fp);
    printf("Decompressed file written to %s\n", output_path);
    return EXIT_SUCCESS;
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
        work.original_size = get_file_size(work.input_path);
        work.result = compress_file(work.input_path, work.output_path);
        work.compressed_size = get_file_size(work.output_path);

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
    char cmd[1024];
    char temp_dir[256];

    // Create temporary directory
    snprintf(temp_dir, sizeof(temp_dir), "temp_extract_%ld", (long)time(NULL));

#ifdef _WIN32
    snprintf(cmd, sizeof(cmd), "mkdir %s", temp_dir);
#else
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", temp_dir);
#endif

    if (system(cmd) != 0)
    {
        fprintf(stderr, "Failed to create temp directory\n");
        return EXIT_FAILURE;
    }

    // Extract archive
    printf("Extracting %s...\n", archive_path);
    snprintf(cmd, sizeof(cmd), "tar -xzf %s -C %s", archive_path, temp_dir);

    if (system(cmd) != 0)
    {
        fprintf(stderr, "Failed to extract %s\n", archive_path);
        return EXIT_FAILURE;
    }

    // Find all .token.npy files
    snprintf(cmd, sizeof(cmd), "find %s -name '*.token.npy' -type f", temp_dir);

    FILE *fp = fopen(cmd, "r");
    if (!fp)
    {
        fprintf(stderr, "Failed to find .token.npy files\n");
        return EXIT_FAILURE;
    }

    char filepath[MAX_FILENAME];
    while (fgets(filepath, sizeof(filepath), fp))
    {
        // Remove newline
        filepath[strcspn(filepath, "\n")] = '\0';

        // Create output path
        char output_path[MAX_FILENAME];
        snprintf(output_path, sizeof(output_path), "%s.cmp", filepath);

        // Add to work queue
        add_work_item(filepath, output_path);
    }

    fclose(fp);

    // Cleanup temp directory later (we'll need the files for processing)
    return EXIT_SUCCESS;
}

//
// Main batch compression function
//
int batch_compress_archives(void)
{
    printf("CommaVQ Batch Processor %s\n", VERSION);
    printf("Checking for required archive files...\n");

    // Check if required files exist
    if (!check_file_exists("data-0000.tar.gz"))
    {
        fprintf(stderr, "ERROR: data-0000.tar.gz not found!\n");
        fprintf(stderr, "Please ensure both data-0000.tar.gz and data-0001.tar.gz exist in the current directory.\n");
        return EXIT_FAILURE;
    }

    if (!check_file_exists("data-0001.tar.gz"))
    {
        fprintf(stderr, "ERROR: data-0001.tar.gz not found!\n");
        fprintf(stderr, "Please ensure both data-0000.tar.gz and data-0001.tar.gz exist in the current directory.\n");
        return EXIT_FAILURE;
    }

    // Get CPU count
    int num_threads = get_cpu_count();
    printf("✓ Both archive files found\n");
    printf("Detected %d CPU cores\n", num_threads);
    printf("Initializing %d worker threads...\n", num_threads);

    // Initialize thread pool
    init_thread_pool(num_threads);

    // Create worker threads
    thread_t *workers = malloc(sizeof(thread_t) * num_threads);
    if (!workers)
    {
        fprintf(stderr, "Failed to allocate memory for worker threads\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < num_threads; i++)
    {
        if (thread_create(&workers[i], worker_thread, NULL) != 0)
        {
            fprintf(stderr, "Failed to create worker thread %d\n", i);
            free(workers);
            return EXIT_FAILURE;
        }
    }

    // Process both archives
    printf("Processing archives...\n");

    if (extract_and_process_archive("data-0000.tar.gz") != EXIT_SUCCESS)
    {
        free(workers);
        return EXIT_FAILURE;
    }

    if (extract_and_process_archive("data-0001.tar.gz") != EXIT_SUCCESS)
    {
        free(workers);
        return EXIT_FAILURE;
    }

    printf("Compressing %d files using %d threads...\n", g_pool.total_files, num_threads);

    // Wait for all work to complete
    wait_for_completion();

    // Signal shutdown and wait for threads
    mutex_lock(&g_pool.queue_mutex);
    g_pool.shutdown = true;
    cond_broadcast(&g_pool.work_available);
    mutex_unlock(&g_pool.queue_mutex);

    for (int i = 0; i < num_threads; i++)
    {
        thread_join(workers[i]);
    }

    free(workers);

    // Calculate compression ratio (like Python version)
    size_t total_original = (size_t)g_pool.total_files * 1200 * 128 * 10 / 8; // 10 bits per token

    // Sum up all compressed file sizes
    size_t total_compressed = 0;
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "find . -name '*.token.npy.cmp' -exec wc -c {} + | tail -n 1");

    FILE *fp = fopen(cmd, "r");
    if (fp)
    {
        fscanf(fp, "%zu", &total_compressed);
        fclose(fp);
    }

    float compression_ratio = (float)total_original / total_compressed;

    printf("\n=== COMPRESSION COMPLETE ===\n");
    printf("Files processed: %d\n", g_pool.total_files);
    printf("Original data: %zu bytes\n", total_original);
    printf("Compressed data: %zu bytes\n", total_compressed);
    printf("Compression ratio: %.1fx\n", compression_ratio);

    // Cleanup
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
