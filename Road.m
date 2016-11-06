% Author: Andy & Ari
% Last Modified: 6th Nov 2016

classdef Road < StateMachine
	
	properties
		id      % unique id of the TrafficSign
	end
	
	
	methods
		
		function obj = Road(id, x, y)
			obj.id = id;
			obj.state.x = x;
			obj.state.y = y;
		end
		
		function pos = position(obj)
			pos = [obj.state.x, obj.state.y];
		end
		
		function obj = step(obj, sensorInput)
		end
		
		function nState = transition(obj, state, sensorInput)
		end
		
		
		function [] = print(obj)
			fprintf('Road id = %d, X = %d, Y = %d', obj.id, obj.state.x, obj.state.y);
			% TODO: find error in printing. seems to be printing array of cars 'sideways'
		end
		
	end
	
end