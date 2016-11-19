clear all; s = RandStream('mcg16807', 'Seed',sum(100*clock)); RandStream.setDefaultStream(s);
clear classes;
addpath(genpath('Bats')); % the Bats toolbox
addpath(genpath('lib')); % toolboxes and functions provided by others
addpath(genpath('test')); % where unit tests are
addpath(genpath('applications')); % where applications are