function filename = newFilename(filename)

	if (exist([filename '.mat'], 'file'))
		i = 1;
		while (exist([filename '-' int2str(i)], 'file'))
			i = i+1 ;
		end
		filename = [filename '-' int2str(i)];
	end
	
end