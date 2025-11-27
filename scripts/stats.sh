wc -l midi_extraction_log.csv # count how many total rows
grep -c "success" midi_extraction_log.csv # count how many success records
grep -c "no_guitar_tracks" midi_extraction_log.csv # count how many no guitar tracks from midis (from songs or parts)
grep -i "_error" midi_extraction_log.csv | wc -l # how many total errors
grep -i "_error" midi_extraction_log.csv # show all errors

# Per artists stats
cut -d',' -f1 midi_extraction_log.csv | sort | uniq -c # how many songs per artist processed
grep "success" midi_extraction_log.csv | cut -d',' -f1 | sort | uniq -c # how many successes per artist
grep "no_guitar_tracks" midi_extraction_log.csv | cut -d',' -f1 | sort | uniq -c # how many no guitar tracks per artist
grep "_error" midi_extraction_log.csv | cut -d',' -f1 | sort | uniq -c # how many errors per artist
cut -d',' -f1,2 midi_extraction_log.csv | sort | uniq | cut -d',' -f1 | sort | uniq -c # count how many songs each artist attempted
awk -F',' '/_error/ && $0 !~ /no_guitar_tracks/ { count[$1]++ }
           END { for (a in count) print count[a], a }' midi_extraction_log.csv # non "no_guitar_tracks" errors


# Error/status distribution
cut -d',' -f5 midi_extraction_log.csv | sort | uniq -c # shows how many parse errors write errors, missing files, etc
cut -d',' -f4 midi_extraction_log.csv | sort | uniq -c # shows by each status

# Songs # all songs with guitar tracks (success only)
grep "success" midi_extraction_log.csv | cut -d',' -f3 | sort