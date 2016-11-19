% Bats Toolbox
% Version 1.0-beta (R2010a) 5-Jun-2011
% http://louis.dorard.me/bats
%
%   (C) Louis Dorard, 2010-2011.
%   This work is licensed under a GNU General Public License
%   http://www.gnu.org/licenses/gpl.html
%   The software is provided "as is", without warranty of any kind, express
%   or implied.
%
%
% Note: class names are capitalised.
%
% 
% Environments
%   Environment           - Abstract super class for reinforcement learning
%                           problems that provides access to the reward function
%   EnvironmentBandit     - Environment for multi-armed bandit problems
%   EnvironmentTS         - Environment for tree search problems
%
% Bandits
%   BAlg                  - Abstract super class for bandit algorithms
%   RandBAlg              - Random bandit algorithm implementation
%   UcbAlg                - Abstract super class for upper confidence bound-type bandit
%                           algorithms
%   UCB                   - UCB-1 for multi-armed bandits (Auer et al. 2002)
%   LinRel                - LinRel for multi-armed bandits (Auer 2002)
%   kRRUcbAlg             - Abstract super class for ucb-type bandit algorithms that 
%                           perform kernel Ridge Regression to determine M
%   KLinRel               - Kernelised LinRel
%   GPB                   - Gaussian Process Upper Confidence Bandits
%                           (Dorard et al. 2009, Srinivas et al. 2010)
%
% Tree Search
%   TreeSearchInterface   - Interface for tree search algorithms
%   Tree                  - Implementation of a tree structure, with methods to operate
%                           on it
%   BanditTS              - Abstract super class for bandit-based tree search algorithms
%   GPTS                  - Gaussian Processes Tree Search (Dorard et al. 2011)
%   BTree                 - Tree structure extended with bandit instances at nodes
%   ManyBanditsTS         - Abstract super class for many-bandits tree search algorithms
%   UCT                   - Implements the Upper Confidence Trees algorithm (Kocsis &
%                           Szepesvari 2006) and the Bandit Algorithm for Smooth Trees
%                           (Coquelin & Munos 2007), which can be used for Hierarchical
%                           Optimistic Optimisation (Bubeck et al. 2009)
%
% Experiments
%   ExperimentsAbstract   - Abstract super class for experiments
% 
% Util
%   covinv                - Computes the inverse of the covariance matrix with Gaussian
%                           additive noise, from scratch or previous cov mat inverse
%   gpr2                  - Same as GPML/GPR but with fixed signal and noise variances
%   subcovinv             - Computes the inverse of the covariance matrix after removing
%                           the first datapoint.
%   TrainingSet           - Simple implementation of a training set
%   
% Unit Tests
%   TestTrainingSet       - Tests adding and removing oldest arms from a training set
%   TestRandBAlg          - Tests base methods of BAlg and its random
%                           implementation
%   TestUCB               - Tests the UCB-1 implementation contained in the UCB class
%   TestGPB               - Tests the GPB (aka GP-UCB) implementation of the kRRUcbAlg
%                           class
%
%
% Sample code to get started:
%
%   rl = rand(1,N); % rewards list (N arms)
%   e = EnvironmentBandit('bernoulli', rl);
%   b = UCB(N);
%   x = b.choose();
%   y = e.play(x);
%   b.train(x, y);