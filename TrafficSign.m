% Author: Andy
% Last Modified: 6th Nov 2016

classdef TrafficSign < StateMachine
    
    properties
        id      % unique id of the TrafficSign
    end
    
    
    methods
        
        function obj = TrafficSign(id, x, y, h)
            obj.id = id;
            obj.state.x = x;
            obj.state.y = y;
            obj.state.h = h;
           
            
            obj.senseMask = ones(5, 5);
        end
        
        function pos = position(obj)
            pos = [obj.state.x, obj.state.y];
        end
        
        function pos = pose(obj)
            pos = [obj.state.x, obj.state.y, obj.state.h];
        end
        
        function obj = step(obj, sensorInput)
            [nState, act] = obj.transition(obj.state, sensorInput);
            
            obj.state = nState;
            obj = action(obj, act);
        end
        
        function [] = print(obj)
            fprintf('Car id = %d, X = %d, Y = %d\n', obj.id, obj.state.x, obj.state.y);
            % TODO: find error in printing. seems to be printing array of cars 'sideways'
        end
       
    end
    
end