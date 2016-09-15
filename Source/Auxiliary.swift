// Auxilliary.swift
//
// Copyright (c) 2014–2015 Mattt Thompson (http://mattt.me)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// MARK: - Auxillary Global Functions

public func abs<T: AccelerateFloatingPoint>(_ x: [T]) -> [T] {
    return T.abs(x)
}

public func floor<T: AccelerateFloatingPoint>(_ x: [T]) -> [T] {
    return T.floor(x)
}

public func ceil<T: AccelerateFloatingPoint>(_ x: [T]) -> [T] {
    return T.ceil(x)
}

public func neg<T: AccelerateFloatingPoint>(_ x: [T]) -> [T] {
    return T.neg(x)
}

public func clip<T: AccelerateFloatingPoint>(_ x: [T], low: T, high: T) -> [T] {
    return T.clip(x, low: low, high: high)
}

public func copysign<T: AccelerateFloatingPoint>(_ magnitude: [T], sign: [T]) -> [T] {
    return T.copysign(magnitude, sign: sign)
}

public func reciprocal<T: AccelerateFloatingPoint>(_ x: [T]) -> [T] {
    return T.rec(x)
}

public func round<T: AccelerateFloatingPoint>(_ x: [T]) -> [T] {
    return T.round(x)
}

public func threshold<T: AccelerateFloatingPoint>(_ x: [T], low: T) -> [T] {
    return T.threshold(x, low: low)
}

public func truncate<T: AccelerateFloatingPoint>(_ x: [T]) -> [T] {
    return T.trunc(x)
}

// MARK: - Auxilliary Array Extension

public extension Array where Element: AccelerateFloatingPoint {

    public func abs() -> [Element] {
        return Element.abs(self)
    }
    
    public func floor() -> [Element] {
        return Element.floor(self)
    }
    
    public func ceil() -> [Element] {
        return Element.ceil(self)
    }
    
    public func negated() -> [Element] {
        return Element.neg(self)
    }
    
    public func clipped(_ low: Element, high: Element) -> [Element] {
        return Element.clip(self, low: low, high: high)
    }
    
    public func copysign(_ sign: [Element]) -> [Element] {
        return Element.copysign(self, sign: sign)
    }
    
    public func reciprocated() -> [Element] {
        return Element.rec(self)
    }
    
    public func rounded() -> [Element] {
        return Element.round(self)
    }
    
    public func thresholded(_ low: Element) -> [Element] {
        return Element.threshold(self, low: low)
    }
    
    public func truncated() -> [Element] {
        return Element.trunc(self)
    }
    
}

// MARK - Operators

prefix operator -

public prefix func -<T: AccelerateFloatingPoint>(value: [T]) -> [T] {
    return T.neg(value)
}
