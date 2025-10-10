# Compile without verbose:
iverilog -o simv *.v
# Compile with verbose:
iverilog -Wall -o simv *.v
# If you have include directories
iverilog -I include_dir -o simv *.v
# If you’re using SystemVerilog features
iverilog -g2012 -o simv *.v

# Run the simulation
vvp simv

# Open the waveform viewer (if you have GTKWave installed)
gtkwave wave.vcd
