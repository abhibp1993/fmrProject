% Author: Ari Goodman and Andy & Abhishek
% Last Modified: 6 Nov 2016

%% Explanation for modification
% Modified by (Abhishek (abhibp1993) @ Nov 6, 2016, 10.20PM)
% 
% The step function, essentially churns the state machine. In case of
% traffic lights, assuming we don't have adaptive ones, will be independent
% of any sensor input. 
% Also, the action function, expects input as Car and actions performed are
% also on Car. It is not a generic function. 

% I feel the following code can be more efficiently written as follows. 
% SM_TrafficLight:
%   States: 2-tuple of (color = {R, Y, G}, wait_time)
%   Action: {R, G, Y}, where R => 'red will go on'. Further, saying NOT
%   'red' become important if we want red light to be OFF. 
%   transition: 
%       1. If 'red' is ON for 4 time-steps, signal becomes 'green'
%       2. If 'green' is ON for 1 time-steps, signal becomes 'yellow'
%       3. If 'yellow' is ON for 1 time-steps, signal becomes 'red'

%% My suggested code: 
classdef TrafficLight < StateMachine
    
    properties
        id
        x
        y
        h
    end
    
    methods
        
        function obj = TrafficLight(id, x, y, h, light)
            obj.id = id;
            obj.x = x;
            obj.y = y;
            obj.h = h;
            obj.state.light = light;
            obj.state.time = 0;
        end
        
        function pos = position(obj)
            pos = [obj.state.x, obj.state.y];
        end
        
        function pos = pose(obj)
            pos = [obj.state.x, obj.state.y, obj.state.h];
        end
        
        function color = step(obj, ~)
            obj.state = obj.transition(obj.state);
            color = obj.state;
        end
        
        function nState = transition(~, state, ~)
            state.time = state.time + 1;
            if isequal(state.light, 'red') && state.time == 4
                state.time = 0;
                state.light = 'green';
                
            elseif isequal(state.light, 'green') && state.time == 1
                state.time = 0;
                state.light = 'yellow';
                
            elseif isequal(state.light, 'yellow') && state.time == 1
                state.time = 0;
                state.light = 'red';
            end
            
            nState = state;
        end
        
        function [] = print(obj)
            fprintf('TL id = %d, X = %d, Y = %d, h = %d, L = %d, color=%s\n', obj.id, obj.x, obj.y, obj.h, obj.state.light);
        end

    end
end

%% Old Code.
% classdef TrafficLight < StateMachine
%     
%     properties
%         id      % unique id of the traffic Light
%     end
%     
%     
%     methods
%         
%         function obj = TrafficLight(id, x, y, h, light)
%             obj.id = id;
%             obj.state.x = x;
%             obj.state.y = y;
%             obj.state.h = h;
%             obj.state.light = light;
%         end
%         
%         function pos = position(obj)
%             pos = [obj.state.x, obj.state.y];
%         end
%         
%         function pos = pose(obj)
%             pos = [obj.state.x, obj.state.y, obj.state.h];
%         end
%         
%         %TODO:
% %         function obj = step(obj, sensorInput)
% %             [nState] = obj.transition(obj.state, sensorInput);
% %             
% %             obj.state = nState;
% %             obj = action(obj, act);
% %         end
%         
%         % I suggest a slight modification might come handy here.. 
%         function output = step(obj, ~)
%             [nState] = obj.transition(obj.state, sensorInput);
%             
%             obj.state = nState;
%             output = nState;
%         end
%         
%         function nState = transition(obj, state, sensorInput)
%             nState = state;
%             if state.light == 'RED'
%                 nState.light = 'GREEN';
%             elseif state.light == 'YELLOW'
%                 nState.light = 'RED';
%             else
%                 nState.light = 'YELLOW';
%             end
%         end
%         
%         function [] = print(obj)
%             fprintf('TL id = %d, X = %d, Y = %d, h = %d, L = %d\n', obj.id, obj.state.x, obj.state.y);
%         end
%        
%     end
%     
% end
