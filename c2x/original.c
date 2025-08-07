// commavq.c

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#define VERSION "v0.1"

//
// Constants based on file format
//
#define HEADER_SIZE 128
#define FRAME_SIZE 256
#define NUM_FRAMES 1200
#define TOTAL_DATA_SIZE (NUM_FRAMES * FRAME_SIZE)
#define TOTAL_FILE_SIZE (HEADER_SIZE + TOTAL_DATA_SIZE)
#define MASK_SIZE (FRAME_SIZE / 8)

//
// Static NumPy header bytes
//
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

// Function prototypes
void print_usage(const char *prog_name);
int compress_file(const char *input_path, const char *output_path);
int decompress_file(const char *input_path, const char *output_path);
void print_version(void);

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
    fprintf(stderr, "  -c <file>  Compress (delta encode) the .token.npy file\n");
    fprintf(stderr, "  -d <file>  Decompress the .token.npy.cmp file\n");
    fprintf(stderr, "  -v         Print version\n");
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
Function Name: compress_file

Description:
    - Compresses the input VDX file and writes the compressed data to the output file.

Parameters:
    - input_path: The path to the input VDX file.
    - output_path: The path to the output compressed file.

Return:
    - 0 on success, or a negative error code on failure.
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

    // Open output file
    FILE *out_fp = fopen(output_path, "wb");
    if (!out_fp)
    {
        perror("Failed to open output file");
        free(data);
        return EXIT_FAILURE;
    }

    // Write keyframe (first frame, full 256 bytes, no header)
    if (fwrite(data, 1, FRAME_SIZE, out_fp) != FRAME_SIZE)
    {
        perror("Failed to write keyframe");
        free(data);
        fclose(out_fp);
        return EXIT_FAILURE;
    }

    // Process delta frames
    uint8_t prev_frame[FRAME_SIZE];
    memcpy(prev_frame, data, FRAME_SIZE);

    for (int f = 1; f < NUM_FRAMES; ++f)
    {
        uint8_t *curr_frame = data + f * FRAME_SIZE;
        uint8_t mask[MASK_SIZE] = {0}; // Bitmask for changes (32 bytes)
        uint8_t changed_bytes[FRAME_SIZE];
        int changed_count = 0;

        for (int i = 0; i < FRAME_SIZE; ++i)
        {
            if (curr_frame[i] != prev_frame[i])
            {
                mask[i / 8] |= (1u << (i % 8));
                changed_bytes[changed_count++] = curr_frame[i];
            }
        }

        // Write mask (32 bytes)
        if (fwrite(mask, 1, MASK_SIZE, out_fp) != MASK_SIZE)
        {
            perror("Failed to write mask");
            free(data);
            fclose(out_fp);
            return EXIT_FAILURE;
        }

        // Write changed bytes (variable length)
        if (fwrite(changed_bytes, 1, changed_count, out_fp) != (size_t)changed_count)
        {
            perror("Failed to write changed bytes");
            free(data);
            fclose(out_fp);
            return EXIT_FAILURE;
        }

        // Update previous frame for next iteration
        memcpy(prev_frame, curr_frame, FRAME_SIZE);
    }

    free(data);
    fclose(out_fp);
    printf("Compressed file written to %s\n", output_path);
    return EXIT_SUCCESS;
}

/*
===============================================================================
Function Name: decompress_file

Description:
    - Decompresses the input VDX file and writes the decompressed data to the output file.

Parameters:
    - input_path: The path to the input compressed file.
    - output_path: The path to the output VDX file.

Return:
    - 0 on success, or a negative error code on failure.
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

    // Allocate output data buffer
    uint8_t *out_data = malloc(TOTAL_DATA_SIZE);
    if (!out_data)
    {
        perror("Failed to allocate output memory");
        free(comp_data);
        return EXIT_FAILURE;
    }

    // Read keyframe
    memcpy(out_data, comp_data, FRAME_SIZE);
    size_t comp_pos = FRAME_SIZE;

    // Reconstruct delta frames
    uint8_t prev_frame[FRAME_SIZE];
    memcpy(prev_frame, out_data, FRAME_SIZE);

    for (int f = 1; f < NUM_FRAMES; ++f)
    {
        if (comp_pos + MASK_SIZE > (size_t)comp_size)
        {
            fprintf(stderr, "Unexpected end of compressed data\n");
            free(comp_data);
            free(out_data);
            return EXIT_FAILURE;
        }

        uint8_t mask[MASK_SIZE];
        memcpy(mask, comp_data + comp_pos, MASK_SIZE);
        comp_pos += MASK_SIZE;

        uint8_t *curr_frame = out_data + f * FRAME_SIZE;
        memcpy(curr_frame, prev_frame, FRAME_SIZE); // Start with previous frame

        int changed_idx = 0;
        for (int i = 0; i < FRAME_SIZE; ++i)
        {
            if (mask[i / 8] & (1u << (i % 8)))
            {
                if (comp_pos >= (size_t)comp_size)
                {
                    fprintf(stderr, "Unexpected end of changed bytes\n");
                    free(comp_data);
                    free(out_data);
                    return EXIT_FAILURE;
                }
                curr_frame[i] = comp_data[comp_pos++];
                changed_idx++;
            }
        }

        // Update previous frame
        memcpy(prev_frame, curr_frame, FRAME_SIZE);
    }

    free(comp_data);

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
// MAIN ENTRY POINT
////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
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