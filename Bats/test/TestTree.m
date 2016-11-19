function TestTree()

clear all;
B = 2;
seed = sum(100*clock);

function createChildren(t, cn)
	maxDepth = 3;
	LC = offspringSum(seed, B, t.features(:,cn));
	for i=1:size(LC,2)-1
		cn2 = t.createNode(cn, LC(:,i));
		if (t.features(1,cn2)<maxDepth), createChildren(t, cn2); end
	end
	cn2 = t.createNode(cn, LC(:,end), 1);
	if (t.features(1,cn2)<maxDepth), createChildren(t, cn2); end
end

t = Tree([0; 1; 0], 4);
cn = 1; % root node
t.createNode(cn, [1; 1; 0]);
t.createNode(cn, [1; 2; 1]);
t.createNode(cn, [1; 3; 0], 1);
% try to create one more node -> should get error
% t.createNode(cn, [1; 4]);
cn = t.nextSibling(t.firstChild(cn)); % should give second child of root
t.createNode(cn, [2; 2; 1], 1);
cn = t.nextSibling(cn); % third child of root
t.createNode(cn, [1; 3; 0]);
t

t = Tree([0; 1; 0], 4);
cn = 1;
createChildren(t, cn);
t

end