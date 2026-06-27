package ocr

import (
	"math"
	"sort"
)

// pt is a 2D point used for geometry (convex hull, min-area rect).
type pt struct {
	X, Y float64
}

// convexHull returns the convex hull (counter-clockwise) of the points using
// Andrew's monotone chain. Input is modified (sorted).
func convexHull(points []pt) []pt {
	n := len(points)
	if n < 3 {
		return points
	}
	sort.Slice(points, func(i, j int) bool {
		if points[i].X != points[j].X {
			return points[i].X < points[j].X
		}
		return points[i].Y < points[j].Y
	})
	cross := func(o, a, b pt) float64 {
		return (a.X-o.X)*(b.Y-o.Y) - (a.Y-o.Y)*(b.X-o.X)
	}
	hull := make([]pt, 0, 2*n)
	// lower hull
	for _, p := range points {
		for len(hull) >= 2 && cross(hull[len(hull)-2], hull[len(hull)-1], p) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, p)
	}
	// upper hull
	lower := len(hull) + 1
	for i := n - 2; i >= 0; i-- {
		p := points[i]
		for len(hull) >= lower && cross(hull[len(hull)-2], hull[len(hull)-1], p) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, p)
	}
	return hull[:len(hull)-1]
}

// minAreaRect computes the minimum-area enclosing rectangle of a convex hull by
// testing each edge orientation (rotating calipers, O(n^2) brute force which is
// fine for the small hulls of text components). Returns the 4 corners.
func minAreaRect(hull []pt) [4]pt {
	if len(hull) < 3 {
		// Degenerate: return axis-aligned bbox of the points.
		return bboxCorners(hull)
	}
	bestArea := math.MaxFloat64
	var best [4]pt
	n := len(hull)
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		ex := hull[j].X - hull[i].X
		ey := hull[j].Y - hull[i].Y
		length := math.Hypot(ex, ey)
		if length < 1e-9 {
			continue
		}
		ux, uy := ex/length, ey/length // edge unit vector
		nx, ny := -uy, ux              // normal

		minU, maxU := math.MaxFloat64, -math.MaxFloat64
		minV, maxV := math.MaxFloat64, -math.MaxFloat64
		for _, p := range hull {
			u := p.X*ux + p.Y*uy
			v := p.X*nx + p.Y*ny
			minU, maxU = math.Min(minU, u), math.Max(maxU, u)
			minV, maxV = math.Min(minV, v), math.Max(maxV, v)
		}
		area := (maxU - minU) * (maxV - minV)
		if area < bestArea {
			bestArea = area
			best = [4]pt{
				{minU*ux + minV*nx, minU*uy + minV*ny},
				{maxU*ux + minV*nx, maxU*uy + minV*ny},
				{maxU*ux + maxV*nx, maxU*uy + maxV*ny},
				{minU*ux + maxV*nx, minU*uy + maxV*ny},
			}
		}
	}
	return orderCorners(best)
}

func bboxCorners(points []pt) [4]pt {
	minX, minY := math.MaxFloat64, math.MaxFloat64
	maxX, maxY := -math.MaxFloat64, -math.MaxFloat64
	for _, p := range points {
		minX, maxX = math.Min(minX, p.X), math.Max(maxX, p.X)
		minY, maxY = math.Min(minY, p.Y), math.Max(maxY, p.Y)
	}
	return [4]pt{{minX, minY}, {maxX, minY}, {maxX, maxY}, {minX, maxY}}
}

// orderCorners returns corners as [topLeft, topRight, bottomRight, bottomLeft]
// and ensures the first edge (TL->TR) is the longer (text) direction.
func orderCorners(c [4]pt) [4]pt {
	cs := c[:]
	sort.Slice(cs, func(i, j int) bool {
		if cs[i].Y != cs[j].Y {
			return cs[i].Y < cs[j].Y
		}
		return cs[i].X < cs[j].X
	})
	top := []pt{cs[0], cs[1]}
	bot := []pt{cs[2], cs[3]}
	sort.Slice(top, func(i, j int) bool { return top[i].X < top[j].X })
	sort.Slice(bot, func(i, j int) bool { return bot[i].X < bot[j].X })
	tl, tr := top[0], top[1]
	bl, br := bot[0], bot[1]
	out := [4]pt{tl, tr, br, bl}

	// Make TL->TR the longer side so recognition width is the text length.
	wLen := math.Hypot(out[1].X-out[0].X, out[1].Y-out[0].Y)
	hLen := math.Hypot(out[3].X-out[0].X, out[3].Y-out[0].Y)
	if hLen > wLen {
		out = [4]pt{out[3], out[0], out[1], out[2]}
	}
	return out
}

// unclipCorners expands a rectangle outward from its center by distance d along
// its own axes (the DBNet "unclip" step, rectangle approximation).
func unclipCorners(c [4]pt, d float64) [4]pt {
	cx := (c[0].X + c[1].X + c[2].X + c[3].X) / 4
	cy := (c[0].Y + c[1].Y + c[2].Y + c[3].Y) / 4
	var out [4]pt
	for i, p := range c {
		dx, dy := p.X-cx, p.Y-cy
		l := math.Hypot(dx, dy)
		if l < 1e-9 {
			out[i] = p
			continue
		}
		out[i] = pt{p.X + dx/l*d, p.Y + dy/l*d}
	}
	return out
}

func cornersBBox(c [4]pt) (minX, minY, maxX, maxY float64) {
	minX, minY = math.MaxFloat64, math.MaxFloat64
	maxX, maxY = -math.MaxFloat64, -math.MaxFloat64
	for _, p := range c {
		minX, maxX = math.Min(minX, p.X), math.Max(maxX, p.X)
		minY, maxY = math.Min(minY, p.Y), math.Max(maxY, p.Y)
	}
	return
}
