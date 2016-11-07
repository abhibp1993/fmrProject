% Author: Andy & Ari & Abhishek 
% Last Modified: 6th Nov 2016

%% Question/Comment (abhibp1993): I do not clearly understand this definition. 
% As per my understanding, Road will not be State Machine, 
% because it doesn't evolve with time or anything. It is static. 
% The utility of this entity in planning is for higher level path planning. 
% Also, the lanes that compose a road provide information about adjacency
% of Lanes, which in turn could be used to define the specifications for
% online-behavior based motion planning. 

% Modified by (abhibp1993 @ Nov 6, 2016, 9.47PM)

%% Code: 1
classdef Road 
	
	properties
		id      % unique id of the TrafficSign
        state   % Patch to maintain the structure. (modified by abhibp1993)
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
		
		function [] = print(obj)
			fprintf('Road id = %d, X = %d, Y = %d', obj.id, obj.state.x, obj.state.y);
			% TODO: find error in printing. seems to be printing array of cars 'sideways'
		end
		
	end
	
end