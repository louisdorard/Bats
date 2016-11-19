%TESTTRAININGSET
% Test of the add and removeOldest methods of training sets.

clear all; s = RandStream('mcg16807', 'Seed',sum(100*clock)); RandStream.setDefaultStream(s);

Tr = TrainingSet();
Tr.add(1,0);
Tr.add(2,0);
Tr.add(3,1);
Tr.add(4,0);

Tr.removeOldest();
Tr.removeOldest();

assertEqual([Tr.X; Tr.Y], [3 4; 1 0]);