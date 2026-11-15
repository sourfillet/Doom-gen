import math
import pytest
from shapely import Polygon,LineString
from wad_parser.portal_contacts import portal_intervals, overlap_length
from wad_parser.navigation import PlayerNavigation

def portal_contact_length(polygon, mouth):
    return sum(high-low for low,high in portal_intervals(polygon,mouth))*mouth.length

def portal_overlap_length(a, b, mouth):
    return overlap_length(portal_intervals(a,mouth),portal_intervals(b,mouth),mouth.length)
from tests.geometry_fixtures import polygon_document

@pytest.mark.parametrize('width',[31,32,40])
@pytest.mark.parametrize('step',[0,24,25])
def test_diagonal_shared_portal(width,step):
    doc=polygon_document([[[(0,0),(0,width),(64+width,width),(64,0)]],
                          [[(64,0),(64+width,width),(192,width),(192,0)]]],start=(32,width/2))
    doc.sectors[1].floor_height=step
    nav=PlayerNavigation(doc)
    assert nav.reaches_point(160,width/2)==(width>=32 and step<=24)
    if step<=24:
        assert {sid for sid,_ in nav.reachable_from([(160,width/2)])}==({0,1} if width>=32 else set())
    # The unchanged >24 drop branch is outside this candidate's contact scope.

@pytest.mark.parametrize('offset',[(0,0),(12000,-16000),(-32000,32000)])
@pytest.mark.parametrize('orientation',[1,-1])
def test_diagonal_contact_roundoff(offset,orientation):
    # Independently derived points on x + 3*y = 80 differ at floating precision.
    def point(t,n):return offset[0]+80-3*t+n,offset[1]+orientation*t
    mouth=LineString([point(0,0),point(24,0)])
    a=Polygon([point(2,0),point(22,0),point(22,32),point(2,32)])
    b=Polygon([point(2,0),point(22,0),point(22,-32),point(2,-32)])
    forward=portal_overlap_length(a,b,mouth)
    assert forward==pytest.approx(20*math.sqrt(10))
    assert forward==portal_overlap_length(b,a,mouth)
    assert forward<=min(portal_contact_length(p,mouth) for p in [a,b])+1e-10
    gap=Polygon([point(2,-1e-7),point(22,-1e-7),point(22,-32),point(2,-32)])
    assert portal_overlap_length(a,gap,mouth)==0

def test_holes_disconnected_intervals_and_tangency():
    mouth=LineString([(0,0),(100,0)])
    a=Polygon([(0,-20),(100,-20),(100,20),(0,20)],
              [[(20,-10),(20,10),(40,10),(40,-10)],[(60,-10),(60,10),(80,10),(80,-10)]])
    assert portal_intervals(a,mouth)==[(0,.2),(.4,.6),(.8,1)]
    assert portal_contact_length(a,mouth)==pytest.approx(60)
    b=Polygon([(20,0),(40,0),(40,-10),(20,-10)])
    assert portal_overlap_length(a,b,mouth)==0
    tangent=Polygon([(50,0),(60,10),(40,10)])
    assert portal_contact_length(tangent,mouth)==0

def test_degenerate_mouth():
    a=Polygon([(0,0),(0,100),(100,100),(100,0)])
    assert portal_contact_length(a,LineString([(50,50),(50,50)]))==0


# Exact free region and mouth from kssht.wad MAP21 (linedef portal at
# (1184,816)-(992,688)). The region is valid, but snapping normals onto the
# portal makes its projection touch itself and GEOS raised "side location
# conflict"; the whole mouth borders the region.
KSSHT_MAP21_REGION = (
    'POLYGON ((931.4285715696144 752.0000001, 943.9999999 752.0000001, 1008.0000001 752.0000001, 1039.9999999 752.0000001, 1039.9999999 783.9999999, 1039.9999999 880.0000001, 1264.0000001 880.0000001, 1295.9999999 880.0000001, 1295.9999999 896, 1557.3333282470703 896, 1584.0000001 896, 1584.0000001 880.0000001, 1615.9999999 880.0000001, 1840.0000001 880.0000001, 1840.0000001 783.9999999, 1840.0000001 752.0000001, 1871.9999999 752.0000001, 1936.0000001 752.0000001, 1948.500000021875 752.0000001, 1924 640, 1912 583, 1859.406844085171 335.9999999, 1840.0000001 335.9999999, 1776.0000001 271.9999999, 1776.0000001 240.00000010000008, 1775.9999999 239.9999999, 1760.0000001 239.9999999, 1760.0000001 224.00000010000008, 1759.9999999 223.99999989999998, 1759.9999999 223.9999999, 1728.0000001 223.9999999, 1712.0000001 207.9999999, 1696.0000002000002 192, 1280 192, 1183.9999997999998 192, 1167.9999999 207.9999999, 1151.9999999 223.9999999, 1120.0000001 223.9999999, 1120.0000001 223.99999989999998, 1119.9999999 224.00000010000008, 1119.9999999 239.9999999, 1104.0000001 239.9999999, 1103.9999999 240.00000010000008, 1103.9999999 271.9999999, 1039.9999999 335.9999999, 1024 335.9999999, 1024 364, 958 628, 955.4285736083984 640, 951 660.6666717529297, 931.4285715696144 752.0000001), (1024 629, 1088 512, 1088 336, 1216 256, 1336 248, 1488 240, 1696 320, 1808 480, 1856 608, 1848 640, 1824 736, 1664 832, 1424 832, 1280 822.3999938964844, 1184 816, 1112 768, 992 688, 1018 640, 1024 629), (1640 846.3999938964844, 1639.1037292480469 846.9377593994141, 1424 832, 1640 846.3999938964844))')


def test_projection_that_touches_itself_still_yields_contact():
    from shapely import from_wkt
    region = from_wkt(KSSHT_MAP21_REGION)
    assert region.is_valid
    assert portal_intervals(region, from_wkt('LINESTRING (1184 816, 992 688)')) == [(0, 1)]
