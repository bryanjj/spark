package org.apache.spark.sql.execution.stat

import org.apache.spark.Logging
import org.apache.spark.sql.{Row, DataFrame}

import scala.util.hashing.MurmurHash3

/**
 * Created by bryan on 7/8/15.
 */
private[sql] object MinHash extends Logging {

  private def rowHash(r: Row, seed: Int) = MurmurHash3.orderedHash(r.toSeq, seed)

  private def minHashIntersectionLength(minHash1: Seq[Int], minHash2: Seq[Int]) = {
    (0 until math.min(minHash1.length, minHash2.length)).foldLeft(0) {
      (sum, k) => if (minHash1(k) == minHash2(k)) sum + 1 else sum
    }
  }

  private[sql] def minHashSimilarity(
      df1: DataFrame,
      df2: DataFrame,
      expectedError: Double = .05) = {

    require(expectedError > 0.0 && expectedError < 1.0, s"error must be between 0.0 and 1.0: $expectedError")

    val numHashes = math.pow(expectedError, -2).toInt

    val kRange = 0 until numHashes
    val zHashes = Seq.fill(numHashes)(Int.MaxValue)

    val seqOp = (hashes: Seq[Int], r: Row) => kRange.map(k => math.min(hashes(k), rowHash(r, k)))
    val combOp = (hashes1: Seq[Int], hashes2: Seq[Int]) => kRange.map(k => math.min(hashes1(k), hashes2(k)))

    val minHash1 = df1.rdd.aggregate(zHashes)(seqOp, combOp)
    val minHash2 = df2.rdd.aggregate(zHashes)(seqOp, combOp)

    minHashIntersectionLength(minHash1, minHash2) / numHashes.toDouble
  }

}
