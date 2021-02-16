rm -f betti.txt

python3 witness_play.py
python3 julia_play.py
julia witness_complex.jl
python3 gudhi_play.py > betti.txt
