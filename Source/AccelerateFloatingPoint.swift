//
//  AccelerateFloatingPoint.swift
//  Surge
//
//  Created by Tyler Fleming Cloutier on 1/18/16.
//  Copyright © 2016 Mattt Thompson. All rights reserved.
//

import Foundation

public protocol AccelerateFloatingPoint: FloatingPoint, ExpressibleByFloatLiteral {
    
    // Operators that Float and Double already implement.
    
    prefix static func -(x: Self) -> Self
    
    static func -(lhs: Self, rhs: Self) -> Self
    
    static func +(lhs: Self, rhs: Self) -> Self
    
    static func *(lhs: Self, rhs: Self) -> Self
    
    static func /(lhs: Self, rhs: Self) -> Self
    
    static func %(lhs: Self, rhs: Self) -> Self
    
    
    static func +=(lhs: inout Self, rhs: Self)
    
    static func -=(lhs: inout Self, rhs: Self)
    
    static func *=(lhs: inout Self, rhs: Self)
    
    static func /=(lhs: inout Self, rhs: Self)
    
    static func %=(lhs: inout Self, rhs: Self)
    
    // MARK: Arithmetic
    
    static func sum(_ x: [Self]) -> Self
    
    static func asum(_ x: [Self]) -> Self
    
    static func max(_ x: [Self]) -> Self
    
    static func min(_ x: [Self]) -> Self
    
    static func mean(_ x: [Self]) -> Self
    
    static func meamg(_ x: [Self]) -> Self
    
    static func measq(_ x: [Self]) -> Self
    
    static func add(_ x: [Self], y: [Self]) -> [Self]
    
    static func sub(_ x: [Self], y: [Self]) -> [Self]
    
    static func mul(_ x: [Self], y: [Self]) -> [Self]

    static func mul(_ x: [Self], y: inout Self) -> [Self]

    static func div(_ x: [Self], y: [Self]) -> [Self]

    static func div(_ x: [Self], y: inout Self) -> [Self]

    static func mod(_ x: [Self], y: [Self]) -> [Self]
    
    static func remainder(_ x: [Self], y: [Self]) -> [Self]
    
    static func sqrt(_ x: [Self]) -> [Self]

    static func square(_ x: [Self]) -> [Self]

    
    // MARK: Vector
    
    static func dot(_ x: [Self], y: [Self]) -> Self
    
    static func cross(_ x: [Self], y: [Self]) -> [Self]
    
    static func length(_ x: [Self]) -> Self
    
    static func norm(_ x: [Self]) -> Self
    
    
    // MARK: Power
    
    static func pow(_ x: [Self], y: [Self]) -> [Self]
    
    
    // MARK: Auxiliary
    
    static func abs(_ x: [Self]) -> [Self]

    static func ceil(_ x: [Self]) -> [Self]
    
    static func clip(_ x: [Self], low: Self, high: Self) -> [Self]

    static func copysign(_ magnitude: [Self], sign: [Self]) -> [Self]
    
    static func floor(_ x: [Self]) -> [Self]
    
    static func neg(_ x: [Self]) -> [Self]
    
    static func rec(_ x: [Self]) -> [Self]
    
    static func round(_ x: [Self]) -> [Self]
    
    static func threshold(_ x: [Self], low: Self) -> [Self]
    
    static func trunc(_ x: [Self]) -> [Self]
    
    
    // MARK: Exponentiation
    
    static func exp(_ x: [Self]) -> [Self]
    
//    static func exp(inout x: [Self])

    static func exp2(_ x: [Self]) -> [Self]

    static func log(_ x: [Self]) -> [Self]
    
    static func log2(_ x: [Self]) -> [Self]
    
    static func log10(_ x: [Self]) -> [Self]
    
    static func logb(_ x: [Self]) -> [Self]
    
    
    // MARK: Fast Fourier Transform
    
    static func fft(_ input: [Self]) -> [Self]
    
    
    // MARK: Trigonometrics

    static func sincos(_ x: [Self]) -> (sin: [Self], cos: [Self])
    
    static func sin(_ x: [Self]) -> [Self]

    static func cos(_ x: [Self]) -> [Self]
    
    static func tan(_ x: [Self]) -> [Self]
    
    static func asin(_ x: [Self]) -> [Self]

    static func acos(_ x: [Self]) -> [Self]

    static func atan(_ x: [Self]) -> [Self]

  static func atan2(_ x: [Self], y: [Self]) -> [Self]

    // MARK: Hyperbolics
    
    static func sinh(_ x: [Self]) -> [Self]
    
    static func cosh(_ x: [Self]) -> [Self]
    
    static func tanh(_ x: [Self]) -> [Self]
    
    static func asinh(_ x: [Self]) -> [Self]
    
    static func acosh(_ x: [Self]) -> [Self]
    
    static func atanh(_ x: [Self]) -> [Self]
    
    
    // MARK: Conversions
    
    static func rad2deg(_ x: [Self]) -> [Self]
    
    static func deg2rad(_ x: [Self]) -> [Self]
    
    // MARK: - Matrix Operations
    
    static func add(_ x: Matrix<Self>, y: Matrix<Self>) -> Matrix<Self>
    
    static func add(_ x: Matrix<Self>, alpha: Self) -> Matrix<Self>
    
    static func sub(_ x: Matrix<Self>, y: Matrix<Self>) -> Matrix<Self>
    
    static func sub(_ x: Matrix<Self>, alpha: Self) -> Matrix<Self>
    
    static func mul(_ x: Matrix<Self>, y: Matrix<Self>) -> Matrix<Self>
    
    static func mul(_ x: Matrix<Self>, alpha: Self) -> Matrix<Self>

    static func div(_ x: Matrix<Self>, y: Matrix<Self>) -> Matrix<Self>
    
    static func div(_ x: Matrix<Self>, alpha: Self) -> Matrix<Self>

    static func inv(_ x: Matrix<Self>) -> Matrix<Self>
    
    static func transpose(_ x: Matrix<Self>) -> Matrix<Self>

}
