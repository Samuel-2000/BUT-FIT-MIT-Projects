{-
  project: flp-fun (1st project regarding decision trees to Functional and Logic Programming course at FIT, BUT)
    author: Samuel Kuchta (xkucht11)
    email: xkucht11@stud.fit.vutbr.cz
    date: 17. 3. 2025
file info: Main. argument parsing, subtask calling.


usage e.g.: ./flp-run -h 
-}



import System.IO.Error (isUserError, ioeGetErrorString)
import System.Exit (exitWith, ExitCode (ExitFailure))
import System.Environment (getArgs)

import Control.Exception (catch, evaluate)
import Control.Monad (forM_)

import DataParser (Value(..), parseData)
import Tree (parseTree, searchTree, createCARTTree)

main :: IO ()
main = do
    -- Parse arguments, get task and execute it
    task <- getArg 0
    case task of 
        "-h"     -> usagePrint
        "--help" -> usagePrint
        "-1"     -> classify
        "-2"     -> train
        _        -> ioError $ userError "Invalid parameter"
    `catch` handleException

-- General exception handler which catches user errors and prints them
-- otherwise it rethrows the error
handleException :: IOError -> IO a
handleException e 
    | isUserError e = do putStrLn (ioeGetErrorString e)
                         exitWith $ ExitFailure 1
    | otherwise = ioError e

usagePrint :: IO ()
usagePrint = do
    putStrLn "Usage: ./flp-run -MODE [OPTIONS ...]"
    putStr "  ./flp-run -1 tree values"
    putStrLn " \n\t run data through decision tree to get labels"
    putStr "  ./flp-run -2 data_train_file"
    putStrLn " \n\t train new decision tree on input data"


getArg :: Int -> IO String
getArg n = do
    args <- getArgs
    if n < 0
        then evaluate (error "Prelude.!!: negative index")
        else case drop n args of
            [] -> ioError $ userError "Insuficient amount of arguments"
            (x:_) -> return x


-- 1. load decision tree and data from files, then classify.
classify :: IO ()
classify = do
    -- Read and parse decision tree
    tree <- getArg 1 >>= readFile >>= either (failWith . ("Tree parse error: " ++)) return . parseTree

    -- Read and parse data
    features <- getArg 2 >>= readFile >>= maybe (ioError $ userError "Data file format error") validateData . parseData False

    -- Classify
    forM_ features $ \feature ->
        maybe (ioError $ userError "Tree index out of range") putStrLn (searchTree tree feature)

    where
        failWith = ioError . userError
        validateData ds = case map features ds of
            [] -> failWith "Empty dataset"
            xs -> evaluate (length xs) >> return xs


-- 2. load training data from file and construct decision tree using Gini indices
train :: IO ()
train = do
    -- Read and parse dataset
    dataset <- getArg 1
        >>= readFile
        >>= evaluate . parseData True
        >>= maybe (ioError $ userError errMsg) return
    
    -- Force evaluation before tree construction
    _ <- evaluate (length dataset)  -- Ensure full dataset is loaded
    print (createCARTTree (-1) dataset)
    where
        errMsg = "Invalid data format: Ensure file contains:\n" ++
                "- Properly formatted numerical values\n" ++
                "- Consistent feature dimensions"