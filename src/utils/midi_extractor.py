import os
import csv
import logging
import argparse
from pathlib import Path
import shutil

import music21 as m21

# --------------------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------------------

DATA_PATH = Path("../../data")
BASE_MUSIC_PATH = DATA_PATH / 'music'
ISOLATE_MIDI_TRACKS = DATA_PATH / 'midi'
LISTS_FOLDER = DATA_PATH / 'lists'
LOGS_FOLDER = DATA_PATH / 'logs'

ALL_ARTISTS_FILE = LISTS_FOLDER / 'artist_names.txt'
TRAINING_ARTISTS_FILE = LISTS_FOLDER / 'training_artist_names.csv'
TRACK_LOG_FILE = LOGS_FOLDER / 'midi_extraction_log.csv'

# --------------------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------------------

def setup_logging():
    LOGS_FOLDER.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_FOLDER / 'midi_extractor.log'

    # basic logging: to console and file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ],
    )

# --------------------------------------------------------------------------------------
# Artist loading
# --------------------------------------------------------------------------------------

def load_training_artists():
    all_artists = []
    with open(ALL_ARTISTS_FILE, 'r', encoding='utf-8') as r:
        lines = r.readlines()
        # TODO: skips first line as it contains a period, fix source file for this
        for line in lines[1:]:
            all_artists.append(line.strip())

    top_artists = []
    with open(TRAINING_ARTISTS_FILE, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:
                top_artists.append(row[0].strip())

    # Intersection of top two lists to find common artists from both lists
    training_artists = [a for a in top_artists if a in all_artists]

    logging.info("Total artists in all artists: %d", len(all_artists))
    logging.info("Total artists in top artists: %d", len(top_artists))
    logging.info("Training artists: %d", len(training_artists))

    return training_artists

# --------------------------------------------------------------------------------------
# CSV log helpers
# --------------------------------------------------------------------------------------

LOG_COLUMNS = [
    'artist',
    'song_file',
    'song_name',
    'status',
    'error_type',
    'error_message'
]

def ensure_log_file():
    """Create log CSV with header if it doesn't exist"""
    LOGS_FOLDER.mkdir(parents=True, exist_ok=True)
    if not TRACK_LOG_FILE.exists():
        with open(TRACK_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS,
                                    quoting=csv.QUOTE_ALL,
                                    escapechar='\\')
            writer.writeheader()

def load_existing_log_status():
    """
    Load existing logs into  dict:
    key = (artists, song_file), value = status
    Used for resume logic.
    """
    status = {}
    if not TRACK_LOG_FILE.exists():
        return status

    with open(TRACK_LOG_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['artist'], row['song_file'])
            status[key] = row['status']
    return status

def log_track_result(artist, song_file, song_name, status, error_type='', error_message=''):
    """Append a result row to the track log"""
    ensure_log_file()
    row = {
        "artist": artist,
        "song_file": song_file,
        "song_name": song_name,
        "status": status,
        "error_type": error_type,
        "error_message": error_message
    }
    with open(TRACK_LOG_FILE, "a", newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS,
                                quoting=csv.QUOTE_ALL,
                                escapechar='\\')
        writer.writerow(row)


# --------------------------------------------------------------------------------------
# MIDI extraction per song
# --------------------------------------------------------------------------------------

def process_song(artist_name, midi_file_name, resume_status, midi_output_dir):
    """
    Process a single MIDI file for a given artis:
    - Parses the MIDI
    Finds guitar parts/writes isolated guitar tracks
    Logs success/failure
    """
    key = (artist_name, midi_file_name)

    # Resume: if we already have a success/no_guitar_tracks entry, skip
    existing_status = resume_status.get(key)

    if existing_status in ("success", "no_guitar_tracks"):
        logging.info(
            "Skipping already processed song: %s | %s (status=%s)",
            artist_name,
            midi_file_name,
            existing_status,
        )
        return

    song_path = BASE_MUSIC_PATH / artist_name / midi_file_name
    song_name = midi_file_name.split(".mid")[0].replace("_"," ")

    if not song_path.exists():
        logging.warning("File does not exist, skipping %s", song_path)
        log_track_result(
            artist=artist_name,
            song_file=midi_file_name,
            song_name=song_name,
            status="missing_file_error",
            error_type="FileNotFoundError",
            error_message="File not found on disk"
        )
        return
    logging.info("Processing %s | %s", artist_name, midi_file_name)

    # 1) Parse MIDI
    try:
        midi = m21.converter.parse(song_path)
        midi.songName = song_name
    except Exception as e:
        logging.exception("Failed to parse MIDI: %s", song_path)
        log_track_result(
            artist=artist_name,
            song_file=midi_file_name,
            song_name=song_name,
            status="parse_error",
            error_type=type(e).__name__, # gets type of the error
            error_message=str(e),
        )
        return

    # 2) Partition by instrument and extract guitar tracks
    wrote_any_guitar = False

    try:
        songs = m21.instrument.partitionByInstrument(midi)
        if songs is None:
            # Some MIDI files may not have instrument parts; treat as no_guitar_tracks
            logging.info("No instrument partition for %s", song_path)
            log_track_result(
                artist=artist_name,
                song_file=-midi_file_name,
                song_name=song_name,
                status="no_guitar_tracks_error"
            )
            return

        for part in songs.parts:
            instruments = part.getElementsByClass(m21.instrument.Instrument)
            if not instruments:
                continue

            for instrument in instruments:
                name = (instrument.instrumentName or "").lower()
                if "guitar" in name or "gtr" in name:
                    guitar_midi_name = f"{song_name} {part.partName or 'Guitar'}.mid"
                    guitar_midi_name = (
                        guitar_midi_name.replace(" ", "_").replace("/","")
                    )
                    output_fp = midi_output_dir / f"{artist_name}_{guitar_midi_name}"

                    try:
                        part.write("midi", fp=output_fp)
                        logging.info("Wrote guitar track: %s", output_fp)
                        wrote_any_guitar = True
                    except Exception as e:
                        logging.exception(
                            "Failed to write guitar MIDI for %s / %s",
                            artist_name,
                            midi_file_name,
                        )
                        # Still continue; we'll capture failure if *no* guitars were successfully written
                        log_track_result(
                            artist=artist_name,
                            song_file=midi_file_name,
                            song_name=song_name,
                            status="write_error",
                            error_type=type(e).__name__,
                            error_message=str(e)
                        )

            if wrote_any_guitar:
                log_track_result(
                    artist=artist_name,
                    song_file=midi_file_name,
                    song_name=song_name,
                    status="success",
                )
            else:
                log_track_result(
                    artist=artist_name,
                    song_file=midi_file_name,
                    song_name=song_name,
                    status="no_guitar_tracks_error"
                )
    except Exception as e:
        logging.exception("Unexpected processing error for %s", song_path)
        log_track_result(
            artist=artist_name,
            song_file=midi_file_name,
            song_name=song_name,
            status="processing_error",
            error_type=type(e).__name__,
            error_message=str(e),
        )

# --------------------------------------------------------------------------------------
# Artist-level processing
# --------------------------------------------------------------------------------------

def process_artist(artist_name, resume_status, midi_output_dir):
    artist_path = BASE_MUSIC_PATH / artist_name
    if not artist_path.exists():
        logging.warning("Artist folder does not exist, skipping: %s", artist_path)
        return

    artist_midi_files = [
        f for f in os.listdir(artist_path) if f.lower().endswith(".mid")
    ]
    logging.info("%d MIDI files found for %s", len(artist_midi_files), artist_name)

    for midi_file_name in artist_midi_files:
        process_song(
            artist_name=artist_name,
            midi_file_name=midi_file_name,
            resume_status=resume_status,
            midi_output_dir=midi_output_dir
        )

# --------------------------------------------------------------------------------------
# CLI & main
# --------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MIDI extractor with logging and retry/resume functionality"
    )

    parser.add_argument(
        "--artist-offset",
        type=int,
        default=0,
        help="Start index in training artist list (for batching)"
    )

    parser.add_argument(
        "--artist-limit",
        type=int,
        default=None,
        help="Number of artists to process from the offset (for batching)",
    )

    parser.add_argument(
        "--artist-name",
        action="append",
        help=(
            "Specific artist name(s) to process"
            "Can be passed multiple times. Overrides offset/limit if provided"
        )
    )

    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing MIDI output folder before running",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from existing log: skip songs already logged as "
            "success or no_guitar_tracks"
        ),
    )
    return parser.parse_args()

def main():
    setup_logging()
    args = parse_args()

    # Prepare output directory
    midi_output_dir = ISOLATE_MIDI_TRACKS
    if args.clean_output and midi_output_dir.exists():
        logging.info("Cleaning MIDI output folder: %s", midi_output_dir)
        shutil.rmtree(midi_output_dir)
    midi_output_dir.mkdir(parents=True, exist_ok=True)

    # Logging CSV
    ensure_log_file()
    resume_status = load_existing_log_status() if args.resume else {}

    # Load artists
    training_artists = load_training_artists()

    # Decide which artists to process
    if args.artist_name:
        # explicit artist names
        artists_to_process = [
            a for a in training_artists if a in set(args.artist_name)
        ]
    else:
        # slice using offset/limit
        start = args.artist_offset # defaults to 0 if not specified
        end = len(training_artists) if args.artist_limit is None else start + args.artist_limit
        artists_to_process = training_artists[start:end]
        # uses full length if no limit specified, else start + limit, so (start..start+limit)

    # main loop
    for artist in artists_to_process:
        logging.info("=== Processing artist: %s === ", artist)
        try:
            process_artist(
                artist_name=artist,
                resume_status=resume_status,
                midi_output_dir=midi_output_dir
            )
        except Exception:
            # Artist-level guard: even if something happens, we move to the next artist
            logging.exception("Unhandled exception while processing artist %s", artist)

    logging.info("MIDI Extraction done")

if __name__ == "__main__":
    main()
