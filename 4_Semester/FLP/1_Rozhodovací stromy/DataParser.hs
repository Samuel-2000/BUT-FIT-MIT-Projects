{-
  project: flp-fun (1st project regarding decision trees to Functional and Logic Programming course at FIT, BUT)
    author: Samuel Kuchta (xkucht11)
    email: xkucht11@stud.fit.vutbr.cz
    date: 17. 3. 2025
file info: Parsing data files
-}

module DataParser (
    Value(..),
    parseData,
    getClasses
) where

import Data.Char (isSpace)


-- in classification, class label is not needed, therefore it is empty
data Value = Value {features :: [Float], label :: String} deriving (Eq)

getClasses :: [Value] -> [String]
getClasses = map label


-- split string with commas to a list of strings divided by commas.
commaSplitter :: String -> [String]
commaSplitter s = reverse $ go s [] []
  where
    go [] acc current = reverse current : acc
    go (c:cs) acc current
      | c == ','  = go cs (reverse current : acc) []
      | otherwise = go cs acc (c : current)
    

parseData :: Bool -> String -> Maybe [Value]
parseData parseClass content
        | null parsedLines || not (allSameFeatureLength parsedLines) = Nothing
        | otherwise = Just parsedLines
    where
        contentLines = filter (not . all isSpace) (lines content)
        parsedLines = map (parseVal parseClass) contentLines
        allSameFeatureLength [] = True
        allSameFeatureLength (x:xs) =
            let targetLen = length (features x)
            in all (\dv -> length (features dv) == targetLen) xs

parseVal :: Bool -> String -> Value
parseVal parseClass line = Value floatList class'
    where
        lineSplit = commaSplitter line
        (class', parts) = if parseClass
                        then let (ps, c) = splitLast lineSplit in (c, ps)
                        else ("", lineSplit)
        floatList = map read parts


splitLast :: [a] -> ([a], a)
splitLast []     = error "splitLast: empty list"
splitLast [x]    = ([], x)
splitLast (x:xs) = let (ys, y) = splitLast xs in (x:ys, y)