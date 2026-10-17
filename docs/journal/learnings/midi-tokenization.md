# MIDI tokenization

MIDI, standing for Musical Instrument Digital Interface, is a digital communication protocol standard in the music sector. It describes the protocol itself, the physical connector to transmit the protocol between devices, and a digital file format. A MIDI file allows to store MIDI messages as a symbolic music file. It is the most abundant among available music datasets. The Lakh dataset as well contains a large corpus of MIDI files.

More over here - [The MIDI protocol - MidiTok's docs](https://miditok.readthedocs.io/en/latest/midi.html) [MIDI.org](https://www.midi.org/)

## **MIDI Messages**

A message expresses an event or an information. It takes the form of a series of bytes. The first is the Status byte which specifies the type of message and the channel, followed by one or two data bytes which contain the information. All the messages and their significations are described in the GM1 and GM2 specifications. The most important are:

- *Note On*: a note is being played, specifies its pitch and velocity;
- *Note Off*: a note is released, specifies the note (by its pitch) to stop and the velocity;
- *Time Signature Change*: indicates the current time signature;
- *Tempo Change*: indicates the current tempo;
- *Program Change*: specifies the current instrument being played;
- *Control Change*: a control parameter is modified or applied. The modulation wheel, foot sustain pedal, volume control or bank select are for instance effects transcribed into Control Change messages.

Note that these messages are “voice messages”, which means that each of them is applied within a channel that is specified in its status byte. The MIDI protocol handles up to sixteen channels which allows to connect multiple devices that are playing and communicating simultaneously. The channel 10 is reserved for drums, which is a specific “program” in which the pitch values corresponds to drum sounds like kicks, snares, or hi-hats.

## **Time in MIDI**

Time in MIDI is determined by its **time division**, which is a clock signal expressed in **ticks per quarter note** (tpq), and can be seen as a time resolution. Common time division values are 384, 480 and 960 tpq as they are divisible by 3, 4, 6 and 8 which are common time signature numerators and denominators. The time division can also be set in ticks per second, but this option is more rarely encountered as it makes less sense to use seconds as the tempo and time signature are known in MIDI. The time division is the first information that can be read at the beginning of a file, and a MIDI file can only have one time division.

The number of ticks per bar and ticks per beat can be calculated from the MIDI’s time division ($`time_{div}`$) and the current time signature ($`\frac{ts_{num}}{ts_{denom}}`$):

- $`tpbeat=time_{div}×\frac{4}{(ts_{denom})}`$
- $`tpbar=tpbeat×ts_{num}`$

Hence, for a $`\frac{4}{4}`$ time signature, the number of ticks per beat is equal to the time division (as a beat is equal to a quarter note) and the number of ticks per bar is equal to four times the number of ticks per beat.
