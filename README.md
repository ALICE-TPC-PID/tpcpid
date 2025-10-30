# tpcpid
Framework for the PID calibration of the ALICE TPC in Run 3 and beyond

Moinsen, ich habe mal eine Struktur ausgedacht. Die Skripte die wir brauchen sind alle noch da, die werde ich dann anpassen.
Mit BasicQA und NeuralNetworkClass habe ich nichts am Hut, die fasse ich nicht an.

Ich werden in "Running" ein bisschen python und bash magic machen, sodass die Richtigen Skripte ausgeführt und gespeichert werden.
Der ganze output wird dann gut sortiert und mit allen hintergrund infos, wie logs etc., in outputs gespeichert. 

Wenn man dann eine production genau angeschaut und untersucht hat, dass alles passt, kann man sie auf "/lustre/alice/tpcpid/" verschieben. 

So wie ich mir das vorstelle, wird sich dann in BasicQA, BBFittingAndQA, NeuralNetworkClass und Training-Neural-Networks nichts verändern. 
In Running werden temporär Jobs erstellt, die dann hinterher gelöscht werden.
Außerdem kann man da natürlich unterschiedliche configuration files speichern.
Der einzige Ordner in dem sich nennenswert Sachen verändern werden, ist "output".