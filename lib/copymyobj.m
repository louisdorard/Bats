% Makes a 'deep' copy of an object, similar to using a copy constructor (as opposed to copying a reference to an object)
%
% Author: Holger Hiebel
% http://www.mathworks.com/matlabcentral/fileexchange/20972-copymyobj

function obj = copymyobj(arg) 
	filename=strcat(pwd,filesep,'temp.mat');
	save(filename,'arg');
	obj=load(filename);
	obj=obj.arg;
	delete(filename);
end