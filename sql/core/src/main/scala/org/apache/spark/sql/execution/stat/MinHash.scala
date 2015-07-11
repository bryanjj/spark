/*
* Licensed to the Apache Software Foundation (ASF) under one or more
* contributor license agreements.  See the NOTICE file distributed with
* this work for additional information regarding copyright ownership.
* The ASF licenses this file to You under the Apache License, Version 2.0
* (the "License"); you may not use this file except in compliance with
* the License.  You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

package org.apache.spark.sql.execution.stat

import org.apache.spark.Logging
import org.apache.spark.sql.{Row, DataFrame}


import scala.util.hashing.MurmurHash3

/**
 * Object for computing the min hash algorithm for dataframe similarity
 * https://en.wikipedia.org/wiki/MinHash
 */
private[sql] object MinHash extends Logging {

  /**
   * Hashes a dataframe row
   * @param row the row to hash
   * @param seed a seed for the hash
   * @return the hash value
   */
  private def rowHash(row: Row, seed: Int) = MurmurHash3.orderedHash(row.toSeq, seed)


  /**
   * get the number of intersections between seqs of min hashes
   * @param minHash1 seq of min hashes
   * @param minHash2 seq of min hashes
   * @return the number of hashes which are the same
   */
  private def minHashIntersectionLength(minHash1: Seq[Int], minHash2: Seq[Int]) = {
    (0 until math.min(minHash1.length, minHash2.length)).foldLeft(0) {
      (sum, k) => if (minHash1(k) == minHash2(k)) sum + 1 else sum
    }
  }

  /**
   * computer the similarity between 2 dataframes using the min hash algorithm with k different hash functions,
   * where k is determined by the expected error.
   * https://en.wikipedia.org/wiki/MinHash
   * @param df1 dataframe 1 to compare
   * @param df2 dataframe 2 to compare
   * @param expectedError value between 0.0 and 1.0 exclusive.
   *                      The smaller the error, the more hash functions will be used.
   *
   * @return
   */
  private[sql] def minHashSimilarity(
      df1: DataFrame,
      df2: DataFrame,
      expectedError: Double) = {

    require(expectedError > 0.0 && expectedError < 1.0, s"error must be between 0.0 and 1.0: $expectedError")

    // number of hash functions, k, is equal to the inverse square of the error
    val numHashes = math.pow(expectedError, -2).toInt

    val kRange = 0 until numHashes
    val zHashes = Seq.fill(numHashes)(Int.MaxValue)

    // aggregate will create k hashes for each row and find the min of each
    val seqOp = (hashes: Seq[Int], r: Row) => kRange.map(k => math.min(hashes(k), rowHash(r, k)))
    val combOp = (hashes1: Seq[Int], hashes2: Seq[Int]) => kRange.map(k => math.min(hashes1(k), hashes2(k)))

    val minHash1 = df1.rdd.aggregate(zHashes)(seqOp, combOp)
    val minHash2 = df2.rdd.aggregate(zHashes)(seqOp, combOp)

    // estimate of similarity is the intersection of min hashes divided by the total number of hashes, k
    minHashIntersectionLength(minHash1, minHash2) / numHashes.toDouble
  }

}
