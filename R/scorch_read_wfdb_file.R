#===============================================================================
# FUNCTION FOR READING WFDB (WAVEFORM DATABASE) FILES
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Read Waveform Database (WFDB) Files
#'
#' @description
#' Reads physiological waveform data stored in WFDB (WaveForm DataBase) format,
#' the standard format used by PhysioNet databases including MIMIC-IV-ECG,
#' PTB-XL, and others. Each WFDB record consists of two files: a \code{.hea}
#' header file (plain text metadata) and a \code{.dat} data file (binary
#' waveform samples).
#'
#' The function reads the header to learn the file's structure (number of
#' signals, sampling frequency, number of samples, binary format, gain, and
#' baseline), then reads and reshapes the binary data accordingly. This
#' header-driven approach means the function works with any standard WFDB
#' record, not just a specific dataset.
#'
#' @param dat_path Character string. Path to the \code{.dat} file containing
#'   binary waveform data.
#'
#' @param hea_path Character string. Path to the \code{.hea} file containing
#'   header metadata.
#'
#' @param validate An optional named list of expected values to check against
#'   the header. If any value does not match, the function returns a soft
#'   failure (\code{success = FALSE}) instead of proceeding. Supported fields:
#'   \describe{
#'     \item{\code{leads}}{Integer. Expected number of signals/leads.}
#'     \item{\code{freq}}{Numeric. Expected sampling frequency in Hz.}
#'     \item{\code{samples}}{Integer. Expected number of samples per signal.}
#'     \item{\code{format}}{Integer. Expected WFDB format code (e.g., 16).}
#'   }
#'   Default is \code{NULL} (no validation; accept whatever the header says).
#'
#'   This is useful when processing a known dataset and you want to reject
#'   files that deviate from the expected specification. For example, when
#'   working with 12-lead ECG data (12 leads, 500 Hz, 5000 samples, format 16):
#'   \preformatted{
#'   validate = list(leads = 12, freq = 500, samples = 5000, format = 16)
#'   }
#'   You can specify any subset of fields. Unspecified fields are not checked.
#'
#' @param verbose Logical. If \code{TRUE} (default), prints progress messages
#'   via \code{message()}.
#'
#' @returns An object of class \code{"scorch_wfdb"} (a list) with:
#'   \describe{
#'     \item{\code{success}}{Logical. \code{TRUE} if the file was read
#'       successfully.}
#'     \item{\code{signals}}{Integer matrix (n_signals x n_samples) containing
#'       raw waveform values. Row names are the signal/lead names from the
#'       header. \code{NULL} if \code{success = FALSE}.}
#'     \item{\code{metadata}}{A list containing:
#'       \describe{
#'         \item{\code{lead_names}}{Character vector of signal names
#'           (e.g., \code{"I"}, \code{"II"}, \code{"V1"}).}
#'         \item{\code{sampling_freq}}{Numeric. Sampling frequency in Hz.}
#'         \item{\code{n_samples}}{Integer. Number of samples per signal.}
#'         \item{\code{n_signals}}{Integer. Number of signals/leads.}
#'         \item{\code{format}}{Integer. WFDB format code used.}
#'         \item{\code{gain}}{Numeric vector. ADC gain for each signal
#'           (ADC units per physical unit).}
#'         \item{\code{baseline}}{Numeric vector. ADC baseline for each
#'           signal.}
#'       }
#'       \code{NULL} if \code{success = FALSE}.}
#'     \item{\code{error}}{Character string with error description if
#'       \code{success = FALSE}. \code{NULL} otherwise.}
#'   }
#'
#' @details
#'
#' \strong{WFDB Format Overview}
#'
#' WFDB is the standard format for PhysioNet's physiological signal databases.
#' The header file (\code{.hea}) is plain text with:
#' \itemize{
#'   \item Line 1: Record name, number of signals, sampling frequency, number
#'     of samples.
#'   \item Lines 2+: One line per signal with format, gain, baseline, and
#'     signal name.
#' }
#'
#' The data file (\code{.dat}) contains binary samples stored in interleaved
#' order: all signals for timepoint 1, then all signals for timepoint 2, etc.
#'
#' \strong{Supported Binary Formats}
#'
#' The function currently supports three WFDB binary formats:
#' \describe{
#'   \item{Format 16}{16-bit signed integer (2 bytes/sample). The most common
#'     format, used by MIMIC-IV-ECG and many PhysioNet databases.}
#'   \item{Format 32}{32-bit signed integer (4 bytes/sample). Used for
#'     high-resolution recordings.}
#'   \item{Format 80}{8-bit unsigned integer with offset (1 byte/sample).
#'     Values are stored as unsigned with an offset of 128, so the effective
#'     range is -128 to +127.}
#' }
#'
#' Format 212 (12-bit packed, 2 samples in 3 bytes) is not yet supported but
#' is planned for a future release.
#'
#' \strong{Soft-Fail Pattern}
#'
#' This function uses a soft-fail pattern: errors return a list with
#' \code{success = FALSE} and a descriptive \code{error} message, rather
#' than throwing via \code{stop()}. This makes it safe for batch processing
#' where one bad file should not crash a loop over thousands of records.
#'
#' \strong{Physical Unit Conversion}
#'
#' The returned \code{signals} matrix contains raw ADC values. To convert to
#' physical units (e.g., millivolts for ECG), use the gain and baseline from
#' metadata:
#' \preformatted{
#' physical <- (raw - baseline) / gain
#' }
#'
#' @examples
#' \dontrun{
#' # Basic usage -- reads whatever the header describes
#' ecg <- scorch_read_wfdb_file("path/to/record.dat", "path/to/record.hea")
#'
#' if (ecg$success) {
#'   dim(ecg$signals)       # e.g., 12 x 5000
#'   ecg$metadata$lead_names # e.g., "I", "II", ..., "V6"
#' }
#'
#' # With validation for expected format
#' ecg <- scorch_read_wfdb_file(
#'   "path/to/record.dat",
#'   "path/to/record.hea",
#'   validate = list(leads = 12, freq = 500, samples = 5000, format = 16)
#' )
#'
#' # Silent batch processing
#' results <- lapply(seq_len(nrow(metadata)), function(i) {
#'   scorch_read_wfdb_file(
#'     dat_path = metadata$filepath_dat[i],
#'     hea_path = metadata$filepath_hea[i],
#'     verbose = FALSE
#'   )
#' })
#' n_success <- sum(sapply(results, `[[`, "success"))
#' message("Loaded ", n_success, " of ", length(results), " files.")
#' }
#'
#' @family data loading
#'
#' @export

scorch_read_wfdb_file <- function(dat_path,
                                  hea_path,
                                  validate = NULL,
                                  verbose = TRUE) {

  # ===== Helper: soft-fail constructor ======================================

  #- Returns a scorch_wfdb object with success = FALSE and descriptive error.
  #- Used throughout for consistent soft-fail returns.

  fail <- function(msg) {

    result <- list(
      success  = FALSE,
      signals  = NULL,
      metadata = NULL,
      error    = msg
    )

    create_scorch_wfdb_class(result)
  }

  # ===== Input validation ===================================================

  if (!is.character(dat_path) || length(dat_path) != 1) {

    return(fail("`dat_path` must be a single character string."))
  }

  if (!is.character(hea_path) || length(hea_path) != 1) {

    return(fail("`hea_path` must be a single character string."))
  }

  if (!is.null(validate) && !is.list(validate)) {

    return(fail("`validate` must be a named list or NULL."))
  }

  # ===== Step 1: File validation ============================================

  #- Check that both required files exist before proceeding.
  #- .hea = plain text header with metadata.
  #- .dat = binary file with interleaved waveform samples.

  if (verbose) message("Checking file paths...")

  if (!file.exists(dat_path)) {

    return(fail(sprintf("Data file not found: %s", dat_path)))
  }

  if (!file.exists(hea_path)) {

    return(fail(sprintf("Header file not found: %s", hea_path)))
  }

  # ===== Step 2: Header parsing =============================================

  #- The .hea file is plain text with one general info line followed by one
  #- line per signal. We parse it to learn everything needed to read the
  #- binary data: number of signals, sampling frequency, number of samples,
  #- binary format, gain, baseline, and signal names.

  if (verbose) message("Parsing header file...")

  header_result <- tryCatch({

    #- Read all lines from the header file.

    header_lines <- readLines(hea_path)

    #- Parse the first line: general recording information.
    #- Format: RecordName NumSignals SamplingFreq NumSamples [Time Date]
    #- Example: "40341634 12 500 5000 ..."

    header_parts <- strsplit(header_lines[1], " ")[[1]]

    n_signals    <- as.integer(header_parts[2])
    sampling_freq <- as.numeric(header_parts[3])
    n_samples    <- as.integer(header_parts[4])

    #- Parse each signal line (lines 2 through n_signals + 1).
    #- Each signal line format:
    #- FileName Format Gain(Baseline)/Units ADCRes ADCZero InitVal Checksum
    #-   BlockSize Description
    #- Example: "40341634.dat 16 1000(0)/mV 16 0 -36 -5765 0 I"

    lead_names <- character(n_signals)
    formats    <- integer(n_signals)
    gains      <- numeric(n_signals)
    baselines  <- numeric(n_signals)

    for (i in seq_len(n_signals)) {

      signal_info <- strsplit(header_lines[i + 1], " ")[[1]]

      #- Position 2: binary format code (e.g., 16, 32, 80).

      formats[i] <- as.integer(signal_info[2])

      #- Position 3: gain string. Two possible formats:
      #-   "gain(baseline)/units" -- e.g., "1000(0)/mV"
      #-   "gain" -- plain number without parentheses.
      #- The gain is the ADC-units-per-physical-unit conversion factor.
      #- The baseline is the ADC value corresponding to 0 physical units.

      gain_str <- signal_info[3]

      if (grepl("\\(", gain_str)) {

        #- Extract gain: everything before the parenthesis.
        gains[i] <- as.numeric(sub("\\(.*", "", gain_str))

        #- Extract baseline: the number inside parentheses.
        baseline_str <- sub(".*\\((.*)\\).*", "\\1", gain_str)
        baselines[i] <- as.numeric(baseline_str)

      } else {

        gains[i] <- as.numeric(gain_str)
        baselines[i] <- if (length(signal_info) >= 4) {
          as.numeric(signal_info[4])
        } else {
          0
        }
      }

      #- Position 9: signal description / lead name (e.g., "I", "II", "V1").

      lead_names[i] <- signal_info[9]
    }

    #- Return parsed header as a list (NULL means success in our pattern).

    list(
      n_signals     = n_signals,
      sampling_freq = sampling_freq,
      n_samples     = n_samples,
      formats       = formats,
      gains         = gains,
      baselines     = baselines,
      lead_names    = lead_names
    )

  }, error = function(e) {

    #- On any parsing error, return the error message as a string.
    #- The calling code checks: if it's a string, it's an error.

    sprintf("Error parsing header file: %s", e$message)
  })

  #- Check if header parsing failed.
  #- A character result means an error message; a list means success.

  if (is.character(header_result)) {

    return(fail(header_result))
  }

  #- Unpack parsed header for use below.

  n_signals     <- header_result$n_signals
  sampling_freq <- header_result$sampling_freq
  n_samples     <- header_result$n_samples
  formats       <- header_result$formats
  gains         <- header_result$gains
  baselines     <- header_result$baselines
  lead_names    <- header_result$lead_names

  if (verbose) {

    message(sprintf("Header parsed: %d signal(s) at %g Hz, %d samples, format %s.",
                    n_signals, sampling_freq, n_samples,
                    paste(unique(formats), collapse = "/")))
  }

  # ===== Step 3: Optional validation ========================================

  #- If the user provided a `validate` list, check each specified field against
  #- what the header reports. This is useful when batch-processing a known
  #- dataset (e.g., MIMIC-IV-ECG) and you want to reject files that deviate
  #- from the expected specification.

  if (!is.null(validate)) {

    if (!is.null(validate$leads) && n_signals != validate$leads) {

      return(fail(sprintf(
        "Validation failed: expected %d leads, header says %d.",
        validate$leads, n_signals)))
    }

    if (!is.null(validate$freq) && sampling_freq != validate$freq) {

      return(fail(sprintf(
        "Validation failed: expected %g Hz, header says %g Hz.",
        validate$freq, sampling_freq)))
    }

    if (!is.null(validate$samples) && n_samples != validate$samples) {

      return(fail(sprintf(
        "Validation failed: expected %d samples, header says %d.",
        validate$samples, n_samples)))
    }

    if (!is.null(validate$format)) {

      bad_formats <- formats[formats != validate$format]

      if (length(bad_formats) > 0) {

        return(fail(sprintf(
          "Validation failed: expected format %d, found format(s) %s.",
          validate$format, paste(unique(bad_formats), collapse = ", "))))
      }
    }
  }

  # ===== Step 4: Determine binary read parameters ===========================

  #- Map the WFDB format code to readBin() parameters.
  #- Currently supported formats:
  #-   16 -- 16-bit signed integer (2 bytes). Most common (MIMIC-IV-ECG, etc.).
  #-   32 -- 32-bit signed integer (4 bytes). High-resolution recordings.
  #-   80 -- 8-bit unsigned integer (1 byte). Offset binary (subtract 128).
  #-
  #- Format 212 (12-bit packed, 2 samples in 3 bytes) is common in PhysioNet

  #- but requires custom bit-unpacking logic. Planned for a future release.

  #- All signals in a record must use the same format (standard WFDB rule for
  #- single .dat files). Check this assumption.

  unique_formats <- unique(formats)

  if (length(unique_formats) != 1) {

    return(fail(sprintf(
      "Mixed formats in a single record are not supported. Found: %s.",
      paste(unique_formats, collapse = ", "))))
  }

  record_format <- unique_formats[1]

  #- Map format code to readBin parameters.

  format_params <- switch(
    as.character(record_format),

    "16" = list(what = "integer", size = 2, signed = TRUE,  offset = 0L),
    "32" = list(what = "integer", size = 4, signed = TRUE,  offset = 0L),
    "80" = list(what = "integer", size = 1, signed = FALSE, offset = 128L),

    #- Unsupported format.
    {
      return(fail(sprintf(
        "Unsupported WFDB format: %d. Supported formats: 16, 32, 80. ",
        record_format,
        "Format 212 is planned for a future release.")))
    }
  )

  # ===== Step 5: File size validation =======================================

  #- Verify the .dat file size matches what the header says.
  #- Expected size = n_signals * n_samples * bytes_per_sample.

  bytes_per_sample <- format_params$size
  expected_size    <- n_signals * n_samples * bytes_per_sample
  actual_size      <- file.info(dat_path)$size

  if (actual_size != expected_size) {

    return(fail(sprintf(
      "File size mismatch: expected %d bytes (%d signals x %d samples x %d bytes), found %d bytes.",
      expected_size, n_signals, n_samples, bytes_per_sample, actual_size)))
  }

  # ===== Step 6: Binary data reading ========================================

  #- Read the .dat file containing the actual waveform data.
  #- Data is stored interleaved: all signals for timepoint 1, then all signals
  #- for timepoint 2, etc. We read the entire file as a flat vector, then
  #- reshape into a matrix where rows = signals, columns = timepoints.

  if (verbose) message("Reading binary waveform data...")

  read_result <- tryCatch({

    #- Open binary connection and guarantee cleanup with on.exit().

    con <- file(dat_path, "rb")
    on.exit(close(con), add = TRUE)

    #- Read all samples as a flat vector.
    #- Total values = n_signals * n_samples.

    raw_data <- readBin(
      con,
      what   = format_params$what,
      n      = n_signals * n_samples,
      size   = format_params$size,
      signed = format_params$signed
    )

    #- Format 80: unsigned 8-bit with offset of 128.
    #- Subtract the offset to get signed values centered around 0.

    if (format_params$offset != 0L) {

      raw_data <- raw_data - format_params$offset
    }

    #- Reshape from interleaved flat vector to matrix.
    #- byrow = FALSE because data is column-major (interleaved):
    #-   [lead1_t1, lead2_t1, ..., leadN_t1, lead1_t2, lead2_t2, ...]
    #- So filling column-by-column gives us rows = leads, cols = timepoints.

    signal_matrix <- matrix(
      raw_data,
      nrow  = n_signals,
      ncol  = n_samples,
      byrow = FALSE
    )

    #- Assign lead names to rows for easier identification.

    rownames(signal_matrix) <- lead_names

    signal_matrix

  }, error = function(e) {

    sprintf("Error reading binary data: %s", e$message)
  })

  #- Check if binary reading failed.

  if (is.character(read_result)) {

    return(fail(read_result))
  }

  signal_matrix <- read_result

  if (verbose) {

    message(sprintf("Successfully read %d x %d signal matrix.",
                    nrow(signal_matrix), ncol(signal_matrix)))
  }

  # ===== Step 7: Assemble output ============================================

  if (verbose) message("Read completed successfully.")

  metadata <- list(
    lead_names    = lead_names,
    sampling_freq = sampling_freq,
    n_samples     = n_samples,
    n_signals     = n_signals,
    format        = record_format,
    gain          = gains,
    baseline      = baselines
  )

  result <- list(
    success  = TRUE,
    signals  = signal_matrix,
    metadata = metadata,
    error    = NULL
  )

  create_scorch_wfdb_class(result)
}

#=== UTILITY FUNCTIONS =========================================================

#--- SCORCH WFDB CLASS --------------------------------------------------------

#' Create a scorch WFDB Class
#'
#' @description
#' Adds the class \code{"scorch_wfdb"} to a given list object.
#'
#' @param obj A list containing WFDB record components.
#'
#' @returns The input object with class attribute set to include
#'   \code{"scorch_wfdb"}.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' wfdb <- list(
#'   success = TRUE,
#'   signals = matrix(1:24, nrow = 2),
#'   metadata = list(lead_names = c("I", "II"), sampling_freq = 500,
#'                   n_samples = 12, n_signals = 2, format = 16,
#'                   gain = c(1000, 1000), baseline = c(0, 0)),
#'   error = NULL
#' )
#' wfdb <- create_scorch_wfdb_class(wfdb)
#' class(wfdb)
#' }

create_scorch_wfdb_class <- function(obj) {

  class(obj) <- c("scorch_wfdb", class(obj))

  obj
}

#=== METHODS ===================================================================

#--- HEAD ----------------------------------------------------------------------

#' Head Method for scorch WFDB
#'
#' @description
#' Returns the first \code{n} timepoints (columns) of the signal matrix from
#' a \code{scorch_wfdb} object. All signals/leads are shown.
#'
#' @param x An object of class \code{"scorch_wfdb"}.
#'
#' @param n Integer. Number of timepoints (columns) to show. Default is 6
#'   (standard R convention).
#'
#' @param ... Additional arguments (currently unused).
#'
#' @returns If \code{x$success} is \code{TRUE}, a matrix with all leads and the
#'   first \code{n} columns. If \code{x$success} is \code{FALSE}, returns
#'   \code{NULL} with a message.
#'
#' @importFrom utils head
#' @export
#'
#' @examples
#' \dontrun{
#' ecg <- scorch_read_wfdb_file("record.dat", "record.hea")
#' head(ecg)       # first 6 timepoints
#' head(ecg, 10)   # first 10 timepoints
#' }

head.scorch_wfdb <- function(x, n = 6L, ...) {

  if (!x$success) {

    message("No signals loaded (success = FALSE). Error: ", x$error)

    return(invisible(NULL))
  }

  n <- min(n, ncol(x$signals))

  x$signals[, seq_len(n), drop = FALSE]
}

#--- PRINT ---------------------------------------------------------------------

#' Print Method for scorch WFDB
#'
#' @description
#' Prints a summary of a \code{scorch_wfdb} object.
#'
#' @param x An object of class \code{"scorch_wfdb"}.
#'
#' @param ... Additional arguments (currently unused).
#'
#' @returns The input object, invisibly.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' ecg <- scorch_read_wfdb_file("record.dat", "record.hea")
#' print(ecg)
#' }

print.scorch_wfdb <- function(x, ...) {

  #- Helper for optional color formatting.

  highlight <- if (requireNamespace("crayon", quietly = TRUE)) {

    crayon::red

  } else {

    identity
  }

  if (!x$success) {

    cat("scorch_wfdb record (FAILED):\n")
    cat(paste0(" * Error: ", highlight(x$error), "\n"))

    return(invisible(x))
  }

  cat("scorch_wfdb ECG recording:\n")

  cat(paste0(" * Leads: ",
             highlight(paste(x$metadata$lead_names, collapse = ", ")),
             "\n"))

  cat(paste0(" * Sampling frequency: ",
             highlight(paste0(x$metadata$sampling_freq, " Hz")),
             "\n"))

  cat(paste0(" * Samples per lead: ",
             highlight(x$metadata$n_samples),
             "\n"))

  cat(paste0(" * Signal matrix: ",
             highlight(paste(dim(x$signals), collapse = " x ")),
             "\n"))

  cat(paste0(" * Format: ",
             highlight(x$metadata$format),
             "\n"))

  invisible(x)
}

#=== END =======================================================================
