{-
  project: flp-fun (1st project regarding decision trees to Functional and Logic Programming course at FIT, BUT)
    author: Samuel Kuchta (xkucht11)
    email: xkucht11@stud.fit.vutbr.cz
    date: 17. 3. 2025
file info: DecisionTree and CART implementation
-}

module Tree (
    createCARTTree,
    DecisionTree(..),
    parseTree, 
    searchTree
) where

import Data.Array (Array, listArray, bounds, (!))
import Data.Char (isSpace)
import Data.List (nub, groupBy, isPrefixOf, sortBy,
                  elemIndex, sort, group, maximumBy)

import qualified Data.Map as Map
import qualified Data.Map.Strict as M

import Data.Maybe (fromMaybe)
import Data.Ord (comparing)

import DataParser (Value(..), getClasses)

import qualified Data.Array as A

 -- Node with index, threshold, left and right subtree, or Leaf with String
data DecisionTree
    = Node Int Float DecisionTree DecisionTree
    | Leaf String
    deriving (Eq)

instance Show DecisionTree where
    show = flip showTree 0

showTree :: DecisionTree -> Int -> String
showTree tree initialLevel = go tree initialLevel ""
    where
        go :: DecisionTree -> Int -> ShowS
        go (Leaf c) level = 
            showString "\n" 
            . showString (indent level) 
            . showString "Leaf: " 
            . shows c
            
        go (Node i v l r) level = 
            (if level > 0 then showString "\n" else id)
            . showString (indent level)
            . showString "Node: "
            . shows i
            . showString ", "
            . shows v
            . go l (level + 1)
            . go r (level + 1)

        indent :: Int -> String
        indent level = replicate (level * 2) ' '




parseTree :: String -> Either String DecisionTree
parseTree rawTreeString
    | null treeLines = Left "Empty decision tree"
    | "Leaf" `isPrefixOf` root =
        if length treeLines /= 1 then
            Left "Incorrectly ended leaf (has elements under it)"
        else 
            Right (Leaf (getLeaf treeString'))
    | otherwise = 
        case splitTree treeString' of 
            Left err -> Left err
            Right (leftSubTree, rightSubTree) -> newNode root leftSubTree rightSubTree
    where 
        treeLines = filter (not . all isSpace) . lines $ rawTreeString
        treeString' = unlines treeLines
        root = head treeLines



getLeaf :: String -> String
getLeaf = unwords . tail . words

splitTree :: String -> Either String (String, String)
splitTree tree = 
    if (length nextLevelIndent /= 2) || (rightChildIdx == -1) then
        Left "Incorrect tree indentation detected"
    else 
        Right (leftSubTree, rightSubTree)
    where 
        tabsFreeTree = map tabToSpace . lines $ tree 
        children = drop 1 tabsFreeTree

        startsWithExactlyTwoSpaces line = length (takeWhile (== ' ') line) == 2
        nextLevelIndent = filter startsWithExactlyTwoSpaces children

        rightChild = last nextLevelIndent

        rightChildIdxRaw = elemIndex rightChild (drop 1 children)
        rightChildIdx = maybe (-1) (+1) rightChildIdxRaw

        leftSubTree = unlines . map (drop 2) . take rightChildIdx $ children 
        rightSubTree = unlines . map (drop 2) . drop rightChildIdx $ children

        -- Helper function to convert tabs to two spaces
        tabToSpace = concatMap (\c -> if c == '\t' then "  " else [c])


newNode :: String -> String -> String -> Either String DecisionTree
newNode root leftSub rightSub = do
    (idx, val) <- getNodeValues root
    left <- parseTree leftSub
    right <- parseTree rightSub
    return $ Node idx val left right

getNodeValues :: String -> Either String (Int, Float)
getNodeValues node = 
    case words node of
        (_:idxPart:valPart:_) -> 
            let idxStr = stripComma idxPart
                parsedIdx = reads idxStr :: [(Int, String)]
                parsedVal = reads valPart :: [(Float, String)]
            in case (parsedIdx, parsedVal) of
                ([(idx, "")], [(val, "")]) -> Right (idx, val)
                _ -> Left "Error parsing index or value from root node"
        _ -> Left "Root node format incorrect"
    where 
        stripComma s = if last s == ',' then init s else s


searchTree :: DecisionTree -> [Float] -> Maybe String
searchTree tree vals = case tree of
    Leaf c -> Just c
    Node idx threshold left right ->
        let arr = A.listArray (0, length vals - 1) vals
        in  if A.inRange (A.bounds arr) idx
            then if arr A.! idx <= threshold
                    then searchTree left vals
                    else searchTree right vals
            else Nothing







-- CART tree --


giniIdx :: (Ord a) => [a] -> Float
giniIdx classes
        | total == 0 = 1.0
        | otherwise = 1.0 - sumSquares
    where
        frequencyMap = M.fromListWith (+) [(x, 1 :: Int) | x <- classes]
        counts = M.elems frequencyMap
        total = fromIntegral (sum counts)
        sumSquares = sum $ map (\c -> (fromIntegral c / total) ** 2) counts



createCARTTree :: Int -> [Value] -> DecisionTree
createCARTTree maxDepth x = 
    if isEndCART maxDepth x then
        Leaf (getMostFreqClass uniqueClasses)
    else 
        Node bestIdx bestVal (createCARTTree (maxDepth - 1) left) (createCARTTree (maxDepth-1) right)
    where 
        classes = getClasses x
        uniqueClasses = nub classes
        baseGI = giniIdx classes
        numFeatures = length (features (head x))
        featuresCount = [0 .. numFeatures - 1]
        candidateFeatures = map (($ x) . getBest baseGI) featuresCount
        bestFeature = foldl (\acc i -> if third i > third acc then i else acc) (0, 0, 0) candidateFeatures
        (bestIdx, bestVal, _) = bestFeature
        left = filter (\dv -> features dv !! bestIdx <= bestVal) x
        right = filter (\dv -> features dv !! bestIdx > bestVal) x



third :: (a, b, c) -> c
third (_,_,z) = z

-- Check if the CART algorithm should end
isEndCART :: Int -> [Value] -> Bool
isEndCART depth x 
    | depth == 0 = True
    | fromSameClass x = True
    | otherwise = False

fromSameClass :: [Value] -> Bool
fromSameClass x = length (nub $ getClasses x) == 1

getMostFreqClass :: (Ord a) => [a] -> a
getMostFreqClass = head . maximumBy (comparing length) . group . sort


-- (Index, threshold, Information gain)
type Feature = (Int, Float, Float)

getBest :: Float -> Int -> [Value] -> Feature
getBest base featureIdx x = 
    if lenThresholds <= 1 then
        (featureIdx, head thresholds, 0)
    else
        (featureIdx, thresholdAvg !! bestFeatureIdx, infGain !! bestFeatureIdx)
    where
          featureClasses = zip (map ((!! featureIdx) . features) x) (getClasses x)
          sortedData = sortBy (comparing fst) featureClasses

          thresholds = map (fst . head) $ groupBy (\a b -> fst a == fst b) sortedData
          lenThresholds = length thresholds

          thresholdAvg = zipWith (\a b -> (a + b) / 2) thresholds (tail thresholds)

          featureValues = map fst sortedData
          n = length featureValues
          featureArr = listArray (0, n-1) featureValues
          cumulativeCounts = scanl (\m (_, cls) -> Map.insertWith (+) cls 1 m) Map.empty sortedData
          totalCounts = last cumulativeCounts

          computeGini :: Map.Map String Int -> Float
          computeGini counts
              | total == 0 = 0
              | otherwise = 1 - sum [ (fromIntegral c / total)^(2::Int) | c <- Map.elems counts ]
              where total = fromIntegral (sum (Map.elems counts))

          infGain = [ base - (leftGini * leftFrac + rightGini * rightFrac)
                    | threshold <- thresholdAvg
                    , let splitIdx = findSplitIndex featureArr threshold
                          leftCounts = cumulativeCounts !! (splitIdx + 1)
                          rightCounts = Map.mergeWithKey (\_ a b -> Just (a - b)) id (const Map.empty) totalCounts leftCounts
                          leftTotal = fromIntegral $ sum (Map.elems leftCounts)
                          rightTotal = fromIntegral $ sum (Map.elems rightCounts)
                          total = leftTotal + rightTotal
                          leftFrac = if total == 0 then 0 else leftTotal / total
                          rightFrac = if total == 0 then 0 else rightTotal / total
                          leftGini = computeGini leftCounts
                          rightGini = computeGini rightCounts
                    ]

          findSplitIndex :: Array Int Float -> Float -> Int
          findSplitIndex arr target = go (bounds arr)
                where
                    go (low, high)
                        | low > high = high
                        | arr ! mid <= target = go (mid + 1, high)
                        | otherwise = go (low, mid - 1)
                        where
                        mid = (low + high) `div` 2

          bestFeatureIdx = fromMaybe (-1) $ elemIndex (maximum infGain) infGain
