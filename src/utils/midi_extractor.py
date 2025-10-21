# TODO: this will ideally be smartly automated in the future, for now, manual work
import os
import shutil

import music21 as m21
from pathlib import Path

# load the list of any artists files
BASE_MUSIC_PATH = '../../data/music/'
ISOLATE_MIDI_TRACKS = '../../data/midi/'
ARTIST_NAME = 'Guns_N_Roses'

midi_folder = Path(ISOLATE_MIDI_TRACKS)

# clean up midi folder before extraction
if midi_folder.exists():
    shutil.rmtree(midi_folder)
midi_folder.mkdir(parents=True, exist_ok=True)

artist_midi_files = [file for file in os.listdir(BASE_MUSIC_PATH + ARTIST_NAME)]
print(len(artist_midi_files), f'{ARTIST_NAME} MIDI files found')

artist_midis = []

for file in artist_midi_files:
    song_name = file.split('.mid')[0].replace('_',' ')
    midi = m21.converter.parse(f'{BASE_MUSIC_PATH}{ARTIST_NAME}/{file}')
    midi.songName = song_name
    artist_midis.append(midi)

    # partitions by instrument, so multiple guitars can possibly be merged
    songs = m21.instrument.partitionByInstrument(midi)
    for part in songs.parts:
        instruments = part.getElementsByClass(m21.instrument.Instrument)
        if instruments:
            first_instrument = instruments[0]
            if "Electric Guitar" in (first_instrument.instrumentName or ""):
                electric_guitar_mid_name = f'{song_name} {part.partName}.mid'.replace(' ','_')
                print("Song - ", electric_guitar_mid_name)
                part.write('midi', fp=f'{ISOLATE_MIDI_TRACKS}{electric_guitar_mid_name}')


    # TODO: figure out how we can isolate by raw MIDI tracks
    # for i, part in enumerate(midi.parts):
    #     instruments = part.getElementsByClass(m21.instrument.Instrument)
    #
    #     if instruments:
    #         print("Instrument", instruments[0])
    #         for instrument in instruments:
    #             print(instrument)
