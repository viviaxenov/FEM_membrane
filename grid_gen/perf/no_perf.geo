// This code was created by pygmsh v6.0.2.
p444 = newp;
Point(p444) = {-7.0, 0.0, 0.0, 0.4};
p445 = newp;
Point(p445) = {-6.5, 0.0, 0.0, 0.4};
p446 = newp;
Point(p446) = {-7.25, 0.43301270189221935, 0.0, 0.4};
p447 = newp;
Point(p447) = {-7.25, -0.4330127018922192, 0.0, 0.4};
l334 = newl;
Circle(l334) = {p445, p444, p446};
l335 = newl;
Circle(l335) = {p446, p444, p447};
l336 = newl;
Circle(l336) = {p447, p444, p445};
ll111 = newll;
Line Loop(ll111) = {l334, l335, l336};
s2 = news;
Plane Surface(s2) = {ll111};
p448 = newp;
Point(p448) = {7.0, 0.0, 0.0, 0.2};
p449 = newp;
Point(p449) = {7.5, 0.0, 0.0, 0.2};
p450 = newp;
Point(p450) = {6.75, 0.43301270189221935, 0.0, 0.2};
p451 = newp;
Point(p451) = {6.75, -0.4330127018922192, 0.0, 0.2};
l337 = newl;
Circle(l337) = {p449, p448, p450};
l338 = newl;
Circle(l338) = {p450, p448, p451};
l339 = newl;
Circle(l339) = {p451, p448, p449};
ll112 = newll;
Line Loop(ll112) = {l337, l338, l339};
p452 = newp;
Point(p452) = {-8, -2, 0, 0.2};
p453 = newp;
Point(p453) = {-8, 2, 0, 0.2};
p454 = newp;
Point(p454) = {8, 2, 0, 0.2};
p455 = newp;
Point(p455) = {8, -2, 0, 0.2};
l340 = newl;
Line(l340) = {p452, p453};
l341 = newl;
Line(l341) = {p453, p454};
l342 = newl;
Line(l342) = {p454, p455};
l343 = newl;
Line(l343) = {p455, p452};
ll113 = newll;
Line Loop(ll113) = {l340, l341, l342, l343};
s3 = news;
Plane Surface(s3) = {ll113,ll111,ll112};
Physical Surface("load") = {s2};
Physical Line("hole") = {l337, l338, l339};
Physical Surface("plate") = {s3};