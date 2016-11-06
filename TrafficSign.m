% Author: Andy & Ari
% Last Modified: 6th Nov 2016

classdef TrafficSign < StateMachine
    
    properties
        id      % unique id of the TrafficSign
        type    % yield 'Yield' or stop sign 'Stop'
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
        
        % TODO
        function obj = step(obj, sensorInput)
            [nState, act] = obj.transition(obj.state, sensorInput);
            
            obj.state = nState;
            obj = action(obj, act);
        end
        
       function nState = transition(obj, state, sensorInput)
            
       end
        
        
        function [] = print(obj)
            fprintf('TrafficSign id = %d, X = %d, Y = %d, h =%d, type=%d\n', obj.id, obj.state.x, obj.state.y, obj.state.h, obj.type);
            % TODO: find error in printing. seems to be printing array of cars 'sideways'
        end
       
    end
    
end