% Author: Ari Goodman and Andy
% Last Modified: 6 Nov 2016

classdef TrafficLight < StateMachine
    
    properties
        id      % unique id of the traffic Light
    end
    
    
    methods
        
        function obj = TrafficLight(id, x, y, h, light)
            obj.id = id;
            obj.state.x = x;
            obj.state.y = y;
            obj.state.h = h;
            obj.state.light = light;
        end
        
        function pos = position(obj)
            pos = [obj.state.x, obj.state.y];
        end
        
        function pos = pose(obj)
            pos = [obj.state.x, obj.state.y, obj.state.h];
        end
        
        %TODO:
        function obj = step(obj, sensorInput)
            [nState] = obj.transition(obj.state, sensorInput);
            
            obj.state = nState;
        end
        
        function nState = transition(obj, state, sensorInput)
            nState = state;
            if state.light == 'RED'
                nState.light = 'GREEN';
            elseif state.light == 'YELLOW'
                nState.light = 'RED';
            else
                nState.light = 'YELLOW';
            end
        end
        
        function [] = print(obj)
            fprintf('TL id = %d, X = %d, Y = %d, h = %d, L = %d\n', obj.id, obj.state.x, obj.state.y);
        end
       
    end
    
end
