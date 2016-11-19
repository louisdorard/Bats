classdef ExperimentsAbstract < handle
%EXPERIMENTSABSTRACT
%   Allows to define problems, agents (bandit algorithms), and experiments that specify how to run a given agent in a given environment.
%   Experiments are run several times in order to deal with the stochasticity of bandit algorithms, by reporting their average performance.
%
% ExperimentsAbstract properties:
%   agent             - type and parameters of a bandit algorithm
%   env               - type and parameters of an environment
%   expe              - specifies an agent, an environment, parameters for the way the agent will interact in this environment, and stores the results of runs of this experiment
%   sortMode          - for a given environmrnt, how should we list the agents with which we experimented, based on their average performance measures? ('descend' or 'ascend')
%
% ExperimentsAbstract methods:
%   addAgent          - add a new agent to experiment with
%   addEnv            - add a new environment to experiment with
%   addExpe           - specify a new experiment based on one of the existing agents and one of the environments
%   runExpe           - run a given experiment for a certain number of times
%   runAll            - run all experiments for a certain number of times
%   save              - save the current object (instance of an implementation of this abstract class) to a file
%   agentParam2String - presents the parameters of an agent as a string, in a human-friendly way
%   envParam2String   - presents the parameters of an environment as a string, in a human-friendly way


    properties (GetAccess='public', SetAccess='protected')
		agent = {}; % struct with two fields: type (i.e. which algo is used), param (given to the algo's constructor)
		env = {}; % same: type and param, but for the environment
		expe = {}; % an experiment is
		% - a reference to an agent and to an environment
		% - a list of parameters that characterise how the agent will interact in this environment
		% - a list of runs (not characteristic of the experiment)
		sortMode = 'descend'; % the bigger the performance measure, the better
    end

	methods (Access='public')
		
		function [ida obj] = addAgent(obj, type, param)
	        a.type = type;
            a.param = param;
			na = length(obj.agent);
			obj.agent{na+1} = a;
			ida = na+1;
        end
		
		function [ide obj] = addEnv(obj, type, param)
			e.type = type;
			e.param = param;
			nenv = length(obj.env);
			obj.env{nenv+1} = e;
			ide = nenv+1;
        end
		
		function [ide obj] = addExpe(obj, varargin)
			ex.ide = varargin{1};
			ex.ida = varargin{2};
			if (length(varargin)>2)
				c = 1;
				for i=3:length(varargin)
					ex.param(c) = varargin{i};
					c = c+1;
				end
			end
			ex.runs = {};
			nex = length(obj.expe);
			obj.expe{nex+1} = ex;
			ide = nex+1;
        end
		
		function obj = runExpe(obj, idex, n)
			% run experiment ide, n times
			for i=1:n
				disp(['   Run ' int2str(i) '...']);
				fprintf('      ');
				seed = sum(clock);
				rand('seed', seed); % start this run with a new seed
				obj = runOne(obj, idex);
				fprintf('\n');
				nr = length(obj.expe{idex}.runs);
				% add the seed and date info to the run we've just done
                obj.expe{idex}.runs{nr}.seed = seed;
				obj.expe{idex}.runs{nr}.date = date;
			end
		end
		
		function obj = runAll(obj, n)
			for i=1:length(obj.expe)
				disp(['Experiment ' int2str(i) '...']);
				obj.runExpe(i, n);
			end
		end
		
		% IDEAS
		% Method to get all experiments with given algorithm?
		% Method to compare two algorithms?

		function save(obj, filename)
			if (nargin<2)
				filename = 'Results';
			end
			filename = newFilename(filename);
			Ex = obj;
			save([filename '.mat'], 'Ex');
			% unix(['zip -r ' filename ' ./']);
		end
		
		function display(obj)
		% specifies how Matlab should display this object when
			
			fprintf('\nAgents:\n');
			fprintf('------------\n\n');
			nba = length(obj.agent);
			if (nba>0)
				for i=1:nba
					fprintf([num2str(i) ': ' obj.agent{i}.type regexprep(obj.agentParam2String(i), '\', '') '\n']);
				end
			end
			fprintf('\n\nEnvironments:\n');
			fprintf('------------\n\n');
			nbe = length(obj.env);
			if (nbe>0)
				for i=1:nbe
					fprintf([num2str(i) ': ' obj.env{i}.type obj.envParam2String(i) '\n']);
					% go through experiments and find those for this problem
					% sort by mean performance
					liste = [];
					results = [];
					for ide = 1:length(obj.expe)
						ex = obj.expe{ide};
						if (ex.ide == i)
							liste = [liste ide];
							nbr = length(ex.runs);
							if (nbr>0)
								for j=1:nbr
									perf(j) = ex.runs{j}.perf;
								end
								results = [results mean(perf)];
							else
								results = [results 0];
							end
						end
					end
					[B IX] = sort(results, obj.sortMode);
					liste = liste(IX);
		            results = results(IX);

					for j=1:length(liste)
		                ide = liste(j);
		                res = results(j);
						ex = obj.expe{ide};
						nbr = length(ex.runs);
						fprintf(['\t * Expe ' int2str(ide) ': ']);
						obj.printExpe(ide);
						fprintf([';\t Agent ' int2str(ex.ida) '\t -> mean perf: ' num2str(res) ' (' int2str(nbr) ' runs)\n']);
					end
					fprintf('\n');
				end
				fprintf('\n');
			end
			
		end
		
	end	

	methods (Abstract, Access='public')
	
		s = agentParam2String(obj, i);
		
		s = envParam2String(obj, i);
	
	end
	
	methods (Abstract, Access='protected')
	
		r = runOne(obj, ex);
		
		printExpe(obj, i);
	
	end
	
	methods (Access='protected')
		
		% function printExpe(obj, i)
		%             expe = obj.expe{i};
		%             fprintf('\n\n');
		% 		disp(['Experiment ' int2str(i) ':']);
		% 		fprintf('------------\n\n');
		%             disp(expe.description);
		% 		obj.printEnv(expe.ide);
		% 		obj.printAgent(expe.ida);
		% 		fprintf('Results:\n');
		% 		nbr = length(expe.runs);
		% 		if (nbr>0)
		% 			for j=1:nbr
		% 				perf(j) = expe.runs{j}.perf;
		% 			end
		% 			mperf = mean(perf);
		% 			disp(['- Mean performance: ' num2str(mperf) ' (' int2str(nbr) ' runs)']);
		% 			disp(['- Detailed results: ' num2str(perf)]);
		% 		end
		%         end
	
    end
    
end