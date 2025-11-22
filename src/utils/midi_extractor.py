# TODO: this will ideally be smartly automated in the future, for now, manual work
import os
import shutil
import csv

import music21 as m21
from pathlib import Path

# load the list of any artists files
DATA_PATH = Path('../../data')
BASE_MUSIC_PATH = DATA_PATH / 'music/'
ISOLATE_MIDI_TRACKS = DATA_PATH / 'midi/'
LISTS_FOLDER = DATA_PATH / 'lists/'

# artist lists
all_artist_list_file = LISTS_FOLDER / "artist_names.txt"
training_artist_list_file = LISTS_FOLDER / "training_artist_names.csv"

all_artists = []
with open(all_artist_list_file, "r") as r:
    lines = r.readlines()
    for line in lines:
        all_artists.append(line.replace("\n",""))
all_artists = all_artists[1:]
print(f"Total artists: {len(all_artists)}")

top_artists = []
with open(training_artist_list_file, "r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        top_artists.append(row[0])
print(f"Total artists: {len(top_artists)}")

training_artists = []
for artist in top_artists:
    if artist in all_artists:
        training_artists.append(artist)
print(f"Training artists: {len(training_artists)}")

midi_folder = Path(ISOLATE_MIDI_TRACKS)

# clean up midi folder before extraction
if midi_folder.exists():
    shutil.rmtree(midi_folder)
midi_folder.mkdir(parents=True, exist_ok=True)

for ARTIST_NAME in training_artists[3:6]:
    artist_midi_files = [file for file in os.listdir(BASE_MUSIC_PATH / f'{ARTIST_NAME}/')]
    print(len(artist_midi_files), f'{ARTIST_NAME} MIDI files found')

    artist_midis = []

    for file in artist_midi_files:
        song_name = file.split('.mid')[0].replace('_',' ')
        midi = m21.converter.parse(f'{BASE_MUSIC_PATH}/{ARTIST_NAME}/{file}')
        midi.songName = song_name
        artist_midis.append(midi)

        # partitions by instrument, so multiple guitars can possibly be merged
        songs = m21.instrument.partitionByInstrument(midi)
        for part in songs.parts:
            instruments = part.getElementsByClass(m21.instrument.Instrument)
            if instruments:
                # instruments holds all possible instruments used to record the current song
                for instrument in instruments:
                    # any guitar track can be used for now as it is difficult whether the track itself is a rhythm/lead/solo/acoustic track
                    if instrument and "guitar" in (instrument.instrumentName or "").lower():
                        guitar_midi_name = f'{song_name} {part.partName}.mid'.replace(' ','_').replace('/','')
                        print("Song - ", guitar_midi_name)
                        part.write('midi', fp=f'{ISOLATE_MIDI_TRACKS}/{ARTIST_NAME}_{guitar_midi_name}')

