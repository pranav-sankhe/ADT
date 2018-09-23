function [ annotations ] = parseSVLAnnotations( svlFile )

% [ annotations ] = parseSVLAnnotations( svlFile )
% Function can read the time instants information exported from
% SonicVisualizer.
%
% Input:
%   svlFile: The complete path to the svl-file
% 
% Output:
%   annotations: a vector containing the time instants in seconds
%

% $Date$ $Revision$ $Author$

% Author: Christian Dittmar (dmr@idmt.fraunhofer.de)
% Created: Jun 2014
% Fraunhofer IDMT. Copyright 2014

dom = xmlread(svlFile);

points = dom.getElementsByTagName('point');
numPoints = points.getLength;

modelAttribs = dom.getElementsByTagName('model').item(0).getAttributes;
for k = 1:modelAttribs.getLength
  currName = modelAttribs.item(k-1).getName;
  currValue = modelAttribs.item(k-1).getValue;
  
  if (strcmp(currName,'sampleRate'))
    sampleRate = str2num(currValue);
  end
end

annotations = zeros(1,numPoints);

for p = 1:numPoints
  
  currItemAttribs = points.item(p-1).getAttributes;
  
  for k = 1:currItemAttribs.getLength
    currName = currItemAttribs.item(k-1).getName;
    currValue = currItemAttribs.item(k-1).getValue;
    
    if (strcmp(currName,'frame'))
      timeInSamples = str2num(currValue);
      timeInSeconds = timeInSamples/sampleRate;
      annotations(p) = timeInSeconds;
    end    
  end
end

end

