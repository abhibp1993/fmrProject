% Author: Andy & Ari
% Last Modified: 6th Nov 2016

%% Question/Comment (abhibp1993):  
% As per my understanding, TrafficSign will not be State Machine, 
% because it doesn't evolve with time or anything. It is static. 

% Modified by (abhibp1993 @ Nov 6, 2016, 10.20PM)

%% Code
classdef TrafficSign
    
    properties
        id      % unique id of the TrafficSign
        type    % yield 'Yield' or stop sign 'Stop'
        state   % Patch to maintain the structure of implemented code (abhibp1993)
    end
    
    
    methods
        
        function obj = TrafficSign(id, x, y, h, type)
            obj.id = id;
            obj.state.x = x;
            obj.state.y = y;
            obj.state.h = h;
            obj.type = type;
        end
        
        function pos = position(obj)
            pos = [obj.state.x, obj.state.y];
        end
        
        function pos = pose(obj)
            pos = [obj.state.x, obj.state.y, obj.state.h];
        end
        
        
        function [] = print(obj)
            fprintf('TrafficSign id = %d, X = %d, Y = %d, h =%d, type=%d\n', obj.id, obj.state.x, obj.state.y, obj.state.h, obj.type);
            % TODO: find error in printing. seems to be printing array of cars 'sideways'
        end
       
    end
    
end