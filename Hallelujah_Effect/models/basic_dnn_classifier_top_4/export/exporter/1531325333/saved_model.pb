Ä
æ%%
:
Add
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
¶
AsString

input"T

output"
Ttype:
	2	
"
	precisionint’’’’’’’’’"

scientificbool( "
shortestbool( "
widthint’’’’’’’’’"
fillstring 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
B
Equal
x"T
y"T
z
"
Ttype:
2	

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
”
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
b
InitializeTableV2
table_handle
keys"Tkey
values"Tval"
Tkeytype"
Tvaltype
:
Less
x"T
y"T
z
"
Ttype:
2	
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	


OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint’’’’’’’’’"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
D
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
¼
SparseToDense
sparse_indices"Tindices
output_shape"Tindices
sparse_values"T
default_value"T

dense"T"
validate_indicesbool("	
Ttype"
Tindicestype:
2	
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
E
Where

input"T	
index	"%
Ttype0
:
2	

&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.8.02v1.8.0-0-g93bc2e20728ÖĀ

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
k
global_step
VariableV2*
shape: *
dtype0	*
_output_shapes
: *
_class
loc:@global_step

global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
_output_shapes
: *
T0	*
_class
loc:@global_step
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
f
PlaceholderPlaceholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’
h
Placeholder_1Placeholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

transform/initNoOp
}
"transform/transform/inputs/tensionPlaceholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

'transform/transform/inputs/tension_copyIdentity"transform/transform/inputs/tension*#
_output_shapes
:’’’’’’’’’*
T0

-transform/transform/scale_by_min_max_16/ShapeShape'transform/transform/inputs/tension_copy*
T0*
_output_shapes
:
z
transform/transform/inputs/lazyPlaceholder*
shape:’’’’’’’’’*
dtype0*#
_output_shapes
:’’’’’’’’’

$transform/transform/inputs/lazy_copyIdentitytransform/transform/inputs/lazy*
T0*#
_output_shapes
:’’’’’’’’’

,transform/transform/scale_by_min_max_8/ShapeShape$transform/transform/inputs/lazy_copy*
T0*
_output_shapes
:

%transform/transform/inputs/engagementPlaceholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

*transform/transform/inputs/engagement_copyIdentity%transform/transform/inputs/engagement*#
_output_shapes
:’’’’’’’’’*
T0

,transform/transform/scale_by_min_max_4/ShapeShape*transform/transform/inputs/engagement_copy*
_output_shapes
:*
T0
~
#transform/transform/inputs/reservedPlaceholder*
shape:’’’’’’’’’*
dtype0*#
_output_shapes
:’’’’’’’’’

(transform/transform/inputs/reserved_copyIdentity#transform/transform/inputs/reserved*
T0*#
_output_shapes
:’’’’’’’’’

-transform/transform/scale_by_min_max_14/ShapeShape(transform/transform/inputs/reserved_copy*
T0*
_output_shapes
:

*transform/transform/inputs/music_pref_rockPlaceholder*
dtype0	*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

/transform/transform/inputs/music_pref_rock_copyIdentity*transform/transform/inputs/music_pref_rock*#
_output_shapes
:’’’’’’’’’*
T0	

,transform/transform/inputs/music_pref_hiphopPlaceholder*
shape:’’’’’’’’’*
dtype0	*#
_output_shapes
:’’’’’’’’’

1transform/transform/inputs/music_pref_hiphop_copyIdentity,transform/transform/inputs/music_pref_hiphop*
T0	*#
_output_shapes
:’’’’’’’’’
~
#transform/transform/inputs/trustingPlaceholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

(transform/transform/inputs/trusting_copyIdentity#transform/transform/inputs/trusting*
T0*#
_output_shapes
:’’’’’’’’’

-transform/transform/scale_by_min_max_18/ShapeShape(transform/transform/inputs/trusting_copy*
_output_shapes
:*
T0

7transform/transform/inputs/music_pref_traditional_irishPlaceholder*
shape:’’’’’’’’’*
dtype0	*#
_output_shapes
:’’’’’’’’’
Æ
<transform/transform/inputs/music_pref_traditional_irish_copyIdentity7transform/transform/inputs/music_pref_traditional_irish*
T0	*#
_output_shapes
:’’’’’’’’’
~
#transform/transform/inputs/outgoingPlaceholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

(transform/transform/inputs/outgoing_copyIdentity#transform/transform/inputs/outgoing*
T0*#
_output_shapes
:’’’’’’’’’

-transform/transform/scale_by_min_max_12/ShapeShape(transform/transform/inputs/outgoing_copy*
T0*
_output_shapes
:
y
transform/transform/inputs/sexPlaceholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’
}
#transform/transform/inputs/sex_copyIdentitytransform/transform/inputs/sex*
T0*#
_output_shapes
:’’’’’’’’’
}
transform/transform/Identity_3Identity#transform/transform/inputs/sex_copy*#
_output_shapes
:’’’’’’’’’*
T0

&transform/transform/inputs/imaginationPlaceholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

+transform/transform/inputs/imagination_copyIdentity&transform/transform/inputs/imagination*
T0*#
_output_shapes
:’’’’’’’’’

,transform/transform/scale_by_min_max_7/ShapeShape+transform/transform/inputs/imagination_copy*
_output_shapes
:*
T0
}
"transform/transform/inputs/nervousPlaceholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

'transform/transform/inputs/nervous_copyIdentity"transform/transform/inputs/nervous*#
_output_shapes
:’’’’’’’’’*
T0

-transform/transform/scale_by_min_max_11/ShapeShape'transform/transform/inputs/nervous_copy*
T0*
_output_shapes
:

/transform/transform/inputs/music_pref_classicalPlaceholder*
dtype0	*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

4transform/transform/inputs/music_pref_classical_copyIdentity/transform/transform/inputs/music_pref_classical*
T0	*#
_output_shapes
:’’’’’’’’’

*transform/transform/inputs/music_pref_jazzPlaceholder*
shape:’’’’’’’’’*
dtype0	*#
_output_shapes
:’’’’’’’’’

/transform/transform/inputs/music_pref_jazz_copyIdentity*transform/transform/inputs/music_pref_jazz*
T0	*#
_output_shapes
:’’’’’’’’’

&transform/transform/inputs/nationalityPlaceholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

+transform/transform/inputs/nationality_copyIdentity&transform/transform/inputs/nationality*
T0*#
_output_shapes
:’’’’’’’’’

transform/transform/Identity_2Identity+transform/transform/inputs/nationality_copy*
T0*#
_output_shapes
:’’’’’’’’’

(transform/transform/inputs/concentrationPlaceholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

-transform/transform/inputs/concentration_copyIdentity(transform/transform/inputs/concentration*
T0*#
_output_shapes
:’’’’’’’’’

,transform/transform/scale_by_min_max_3/ShapeShape-transform/transform/inputs/concentration_copy*
_output_shapes
:*
T0

+transform/transform/inputs/music_pref_dancePlaceholder*
dtype0	*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

0transform/transform/inputs/music_pref_dance_copyIdentity+transform/transform/inputs/music_pref_dance*#
_output_shapes
:’’’’’’’’’*
T0	

.transform/transform/inputs/hearing_impairmentsPlaceholder*
dtype0	*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

3transform/transform/inputs/hearing_impairments_copyIdentity.transform/transform/inputs/hearing_impairments*#
_output_shapes
:’’’’’’’’’*
T0	

.transform/transform/inputs/hallelujah_reactionPlaceholder*
dtype0	*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

3transform/transform/inputs/hallelujah_reaction_copyIdentity.transform/transform/inputs/hallelujah_reaction*
T0	*#
_output_shapes
:’’’’’’’’’
|
!transform/transform/inputs/stressPlaceholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

&transform/transform/inputs/stress_copyIdentity!transform/transform/inputs/stress*
T0*#
_output_shapes
:’’’’’’’’’

-transform/transform/scale_by_min_max_15/ShapeShape&transform/transform/inputs/stress_copy*
T0*
_output_shapes
:

'transform/transform/inputs/like_dislikePlaceholder*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’*
dtype0

,transform/transform/inputs/like_dislike_copyIdentity'transform/transform/inputs/like_dislike*
T0*#
_output_shapes
:’’’’’’’’’

,transform/transform/scale_by_min_max_9/ShapeShape,transform/transform/inputs/like_dislike_copy*
T0*
_output_shapes
:
~
#transform/transform/inputs/languagePlaceholder*
shape:’’’’’’’’’*
dtype0*#
_output_shapes
:’’’’’’’’’

(transform/transform/inputs/language_copyIdentity#transform/transform/inputs/language*
T0*#
_output_shapes
:’’’’’’’’’

transform/transform/IdentityIdentity(transform/transform/inputs/language_copy*
T0*#
_output_shapes
:’’’’’’’’’
{
 transform/transform/inputs/faultPlaceholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

%transform/transform/inputs/fault_copyIdentity transform/transform/inputs/fault*
T0*#
_output_shapes
:’’’’’’’’’

,transform/transform/scale_by_min_max_6/ShapeShape%transform/transform/inputs/fault_copy*
_output_shapes
:*
T0
j
#transform/transform/inputs/age_copyIdentityPlaceholder*
T0*#
_output_shapes
:’’’’’’’’’

,transform/transform/scale_by_min_max_1/ShapeShape#transform/transform/inputs/age_copy*
_output_shapes
:*
T0
y
transform/transform/inputs/agePlaceholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

,transform/transform/inputs/musical_expertisePlaceholder*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’*
dtype0

1transform/transform/inputs/musical_expertise_copyIdentity,transform/transform/inputs/musical_expertise*
T0*#
_output_shapes
:’’’’’’’’’

-transform/transform/scale_by_min_max_10/ShapeShape1transform/transform/inputs/musical_expertise_copy*
_output_shapes
:*
T0

&transform/transform/inputs/familiarityPlaceholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

+transform/transform/inputs/familiarity_copyIdentity&transform/transform/inputs/familiarity*#
_output_shapes
:’’’’’’’’’*
T0

,transform/transform/scale_by_min_max_5/ShapeShape+transform/transform/inputs/familiarity_copy*
T0*
_output_shapes
:

)transform/transform/inputs/music_pref_popPlaceholder*
dtype0	*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

.transform/transform/inputs/music_pref_pop_copyIdentity)transform/transform/inputs/music_pref_pop*
T0	*#
_output_shapes
:’’’’’’’’’
~
#transform/transform/inputs/thoroughPlaceholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

(transform/transform/inputs/thorough_copyIdentity#transform/transform/inputs/thorough*#
_output_shapes
:’’’’’’’’’*
T0

-transform/transform/scale_by_min_max_17/ShapeShape(transform/transform/inputs/thorough_copy*
T0*
_output_shapes
:

*transform/transform/inputs/music_pref_nonePlaceholder*
shape:’’’’’’’’’*
dtype0	*#
_output_shapes
:’’’’’’’’’

/transform/transform/inputs/music_pref_none_copyIdentity*transform/transform/inputs/music_pref_none*
T0	*#
_output_shapes
:’’’’’’’’’

*transform/transform/inputs/music_pref_folkPlaceholder*
dtype0	*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

/transform/transform/inputs/music_pref_folk_copyIdentity*transform/transform/inputs/music_pref_folk*#
_output_shapes
:’’’’’’’’’*
T0	
~
#transform/transform/inputs/locationPlaceholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

(transform/transform/inputs/location_copyIdentity#transform/transform/inputs/location*#
_output_shapes
:’’’’’’’’’*
T0

transform/transform/Identity_1Identity(transform/transform/inputs/location_copy*#
_output_shapes
:’’’’’’’’’*
T0

+transform/transform/inputs/music_pref_worldPlaceholder*
dtype0	*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

0transform/transform/inputs/music_pref_world_copyIdentity+transform/transform/inputs/music_pref_world*
T0	*#
_output_shapes
:’’’’’’’’’
q
(transform/transform/inputs/activity_copyIdentityPlaceholder_1*#
_output_shapes
:’’’’’’’’’*
T0

*transform/transform/scale_by_min_max/ShapeShape(transform/transform/inputs/activity_copy*
T0*
_output_shapes
:
~
#transform/transform/inputs/activityPlaceholder*
shape:’’’’’’’’’*
dtype0*#
_output_shapes
:’’’’’’’’’

%transform/transform/inputs/positivityPlaceholder*
shape:’’’’’’’’’*
dtype0*#
_output_shapes
:’’’’’’’’’

*transform/transform/inputs/positivity_copyIdentity%transform/transform/inputs/positivity*#
_output_shapes
:’’’’’’’’’*
T0

-transform/transform/scale_by_min_max_13/ShapeShape*transform/transform/inputs/positivity_copy*
T0*
_output_shapes
:
~
#transform/transform/inputs/artisticPlaceholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’

(transform/transform/inputs/artistic_copyIdentity#transform/transform/inputs/artistic*#
_output_shapes
:’’’’’’’’’*
T0

,transform/transform/scale_by_min_max_2/ShapeShape(transform/transform/inputs/artistic_copy*
T0*
_output_shapes
:
{
6transform/transform/scale_by_min_max/min_and_max/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ė
4transform/transform/scale_by_min_max/min_and_max/subSub6transform/transform/scale_by_min_max/min_and_max/sub/x(transform/transform/inputs/activity_copy*
T0*#
_output_shapes
:’’’’’’’’’

Atransform/transform/scale_by_min_max/min_and_max/amax/PlaceholderPlaceholder*
dtype0*
_output_shapes
: *
shape: 

Ctransform/transform/scale_by_min_max/min_and_max/amax/Placeholder_1Placeholder*
dtype0*
_output_shapes
: *
shape: 
}
8transform/transform/scale_by_min_max/min_and_max/sub_1/xConst*
_output_shapes
: *
valueB
 *    *
dtype0
v
1transform/transform/scale_by_min_max/Fill_1/valueConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ą
+transform/transform/scale_by_min_max/Fill_1Fill*transform/transform/scale_by_min_max/Shape1transform/transform/scale_by_min_max/Fill_1/value*
T0*#
_output_shapes
:’’’’’’’’’
o
*transform/transform/scale_by_min_max/mul/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
*transform/transform/scale_by_min_max/add/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
}
8transform/transform/scale_by_min_max_1/min_and_max/sub/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ź
6transform/transform/scale_by_min_max_1/min_and_max/subSub8transform/transform/scale_by_min_max_1/min_and_max/sub/x#transform/transform/inputs/age_copy*#
_output_shapes
:’’’’’’’’’*
T0

Ctransform/transform/scale_by_min_max_1/min_and_max/amax/PlaceholderPlaceholder*
dtype0*
_output_shapes
: *
shape: 

Etransform/transform/scale_by_min_max_1/min_and_max/amax/Placeholder_1Placeholder*
dtype0*
_output_shapes
: *
shape: 

:transform/transform/scale_by_min_max_1/min_and_max/sub_1/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
x
3transform/transform/scale_by_min_max_1/Fill_1/valueConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ę
-transform/transform/scale_by_min_max_1/Fill_1Fill,transform/transform/scale_by_min_max_1/Shape3transform/transform/scale_by_min_max_1/Fill_1/value*#
_output_shapes
:’’’’’’’’’*
T0
q
,transform/transform/scale_by_min_max_1/mul/yConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
q
,transform/transform/scale_by_min_max_1/add/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
}
8transform/transform/scale_by_min_max_2/min_and_max/sub/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ļ
6transform/transform/scale_by_min_max_2/min_and_max/subSub8transform/transform/scale_by_min_max_2/min_and_max/sub/x(transform/transform/inputs/artistic_copy*
T0*#
_output_shapes
:’’’’’’’’’

Ctransform/transform/scale_by_min_max_2/min_and_max/amax/PlaceholderPlaceholder*
dtype0*
_output_shapes
: *
shape: 

Etransform/transform/scale_by_min_max_2/min_and_max/amax/Placeholder_1Placeholder*
shape: *
dtype0*
_output_shapes
: 

:transform/transform/scale_by_min_max_2/min_and_max/sub_1/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
x
3transform/transform/scale_by_min_max_2/Fill_1/valueConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ę
-transform/transform/scale_by_min_max_2/Fill_1Fill,transform/transform/scale_by_min_max_2/Shape3transform/transform/scale_by_min_max_2/Fill_1/value*
T0*#
_output_shapes
:’’’’’’’’’
q
,transform/transform/scale_by_min_max_2/mul/yConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
q
,transform/transform/scale_by_min_max_2/add/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
}
8transform/transform/scale_by_min_max_3/min_and_max/sub/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ō
6transform/transform/scale_by_min_max_3/min_and_max/subSub8transform/transform/scale_by_min_max_3/min_and_max/sub/x-transform/transform/inputs/concentration_copy*
T0*#
_output_shapes
:’’’’’’’’’

Ctransform/transform/scale_by_min_max_3/min_and_max/amax/PlaceholderPlaceholder*
shape: *
dtype0*
_output_shapes
: 

Etransform/transform/scale_by_min_max_3/min_and_max/amax/Placeholder_1Placeholder*
dtype0*
_output_shapes
: *
shape: 

:transform/transform/scale_by_min_max_3/min_and_max/sub_1/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
x
3transform/transform/scale_by_min_max_3/Fill_1/valueConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ę
-transform/transform/scale_by_min_max_3/Fill_1Fill,transform/transform/scale_by_min_max_3/Shape3transform/transform/scale_by_min_max_3/Fill_1/value*
T0*#
_output_shapes
:’’’’’’’’’
q
,transform/transform/scale_by_min_max_3/mul/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
q
,transform/transform/scale_by_min_max_3/add/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
}
8transform/transform/scale_by_min_max_4/min_and_max/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ń
6transform/transform/scale_by_min_max_4/min_and_max/subSub8transform/transform/scale_by_min_max_4/min_and_max/sub/x*transform/transform/inputs/engagement_copy*#
_output_shapes
:’’’’’’’’’*
T0

Ctransform/transform/scale_by_min_max_4/min_and_max/amax/PlaceholderPlaceholder*
shape: *
dtype0*
_output_shapes
: 

Etransform/transform/scale_by_min_max_4/min_and_max/amax/Placeholder_1Placeholder*
_output_shapes
: *
shape: *
dtype0

:transform/transform/scale_by_min_max_4/min_and_max/sub_1/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
x
3transform/transform/scale_by_min_max_4/Fill_1/valueConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ę
-transform/transform/scale_by_min_max_4/Fill_1Fill,transform/transform/scale_by_min_max_4/Shape3transform/transform/scale_by_min_max_4/Fill_1/value*#
_output_shapes
:’’’’’’’’’*
T0
q
,transform/transform/scale_by_min_max_4/mul/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
q
,transform/transform/scale_by_min_max_4/add/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
}
8transform/transform/scale_by_min_max_5/min_and_max/sub/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ņ
6transform/transform/scale_by_min_max_5/min_and_max/subSub8transform/transform/scale_by_min_max_5/min_and_max/sub/x+transform/transform/inputs/familiarity_copy*
T0*#
_output_shapes
:’’’’’’’’’

Ctransform/transform/scale_by_min_max_5/min_and_max/amax/PlaceholderPlaceholder*
shape: *
dtype0*
_output_shapes
: 

Etransform/transform/scale_by_min_max_5/min_and_max/amax/Placeholder_1Placeholder*
dtype0*
_output_shapes
: *
shape: 

:transform/transform/scale_by_min_max_5/min_and_max/sub_1/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
x
3transform/transform/scale_by_min_max_5/Fill_1/valueConst*
_output_shapes
: *
valueB
 *   ?*
dtype0
Ę
-transform/transform/scale_by_min_max_5/Fill_1Fill,transform/transform/scale_by_min_max_5/Shape3transform/transform/scale_by_min_max_5/Fill_1/value*
T0*#
_output_shapes
:’’’’’’’’’
q
,transform/transform/scale_by_min_max_5/mul/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
q
,transform/transform/scale_by_min_max_5/add/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
}
8transform/transform/scale_by_min_max_6/min_and_max/sub/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ģ
6transform/transform/scale_by_min_max_6/min_and_max/subSub8transform/transform/scale_by_min_max_6/min_and_max/sub/x%transform/transform/inputs/fault_copy*
T0*#
_output_shapes
:’’’’’’’’’

Ctransform/transform/scale_by_min_max_6/min_and_max/amax/PlaceholderPlaceholder*
_output_shapes
: *
shape: *
dtype0

Etransform/transform/scale_by_min_max_6/min_and_max/amax/Placeholder_1Placeholder*
dtype0*
_output_shapes
: *
shape: 

:transform/transform/scale_by_min_max_6/min_and_max/sub_1/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
x
3transform/transform/scale_by_min_max_6/Fill_1/valueConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ę
-transform/transform/scale_by_min_max_6/Fill_1Fill,transform/transform/scale_by_min_max_6/Shape3transform/transform/scale_by_min_max_6/Fill_1/value*#
_output_shapes
:’’’’’’’’’*
T0
q
,transform/transform/scale_by_min_max_6/mul/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
q
,transform/transform/scale_by_min_max_6/add/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
}
8transform/transform/scale_by_min_max_7/min_and_max/sub/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ņ
6transform/transform/scale_by_min_max_7/min_and_max/subSub8transform/transform/scale_by_min_max_7/min_and_max/sub/x+transform/transform/inputs/imagination_copy*
T0*#
_output_shapes
:’’’’’’’’’

Ctransform/transform/scale_by_min_max_7/min_and_max/amax/PlaceholderPlaceholder*
dtype0*
_output_shapes
: *
shape: 

Etransform/transform/scale_by_min_max_7/min_and_max/amax/Placeholder_1Placeholder*
shape: *
dtype0*
_output_shapes
: 

:transform/transform/scale_by_min_max_7/min_and_max/sub_1/xConst*
_output_shapes
: *
valueB
 *    *
dtype0
x
3transform/transform/scale_by_min_max_7/Fill_1/valueConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
Ę
-transform/transform/scale_by_min_max_7/Fill_1Fill,transform/transform/scale_by_min_max_7/Shape3transform/transform/scale_by_min_max_7/Fill_1/value*
T0*#
_output_shapes
:’’’’’’’’’
q
,transform/transform/scale_by_min_max_7/mul/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
q
,transform/transform/scale_by_min_max_7/add/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
}
8transform/transform/scale_by_min_max_8/min_and_max/sub/xConst*
_output_shapes
: *
valueB
 *    *
dtype0
Ė
6transform/transform/scale_by_min_max_8/min_and_max/subSub8transform/transform/scale_by_min_max_8/min_and_max/sub/x$transform/transform/inputs/lazy_copy*
T0*#
_output_shapes
:’’’’’’’’’

Ctransform/transform/scale_by_min_max_8/min_and_max/amax/PlaceholderPlaceholder*
dtype0*
_output_shapes
: *
shape: 

Etransform/transform/scale_by_min_max_8/min_and_max/amax/Placeholder_1Placeholder*
dtype0*
_output_shapes
: *
shape: 

:transform/transform/scale_by_min_max_8/min_and_max/sub_1/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
x
3transform/transform/scale_by_min_max_8/Fill_1/valueConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ę
-transform/transform/scale_by_min_max_8/Fill_1Fill,transform/transform/scale_by_min_max_8/Shape3transform/transform/scale_by_min_max_8/Fill_1/value*
T0*#
_output_shapes
:’’’’’’’’’
q
,transform/transform/scale_by_min_max_8/mul/yConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
q
,transform/transform/scale_by_min_max_8/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
}
8transform/transform/scale_by_min_max_9/min_and_max/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ó
6transform/transform/scale_by_min_max_9/min_and_max/subSub8transform/transform/scale_by_min_max_9/min_and_max/sub/x,transform/transform/inputs/like_dislike_copy*#
_output_shapes
:’’’’’’’’’*
T0

Ctransform/transform/scale_by_min_max_9/min_and_max/amax/PlaceholderPlaceholder*
dtype0*
_output_shapes
: *
shape: 

Etransform/transform/scale_by_min_max_9/min_and_max/amax/Placeholder_1Placeholder*
shape: *
dtype0*
_output_shapes
: 

:transform/transform/scale_by_min_max_9/min_and_max/sub_1/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
x
3transform/transform/scale_by_min_max_9/Fill_1/valueConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ę
-transform/transform/scale_by_min_max_9/Fill_1Fill,transform/transform/scale_by_min_max_9/Shape3transform/transform/scale_by_min_max_9/Fill_1/value*#
_output_shapes
:’’’’’’’’’*
T0
q
,transform/transform/scale_by_min_max_9/mul/yConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
q
,transform/transform/scale_by_min_max_9/add/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
~
9transform/transform/scale_by_min_max_10/min_and_max/sub/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ś
7transform/transform/scale_by_min_max_10/min_and_max/subSub9transform/transform/scale_by_min_max_10/min_and_max/sub/x1transform/transform/inputs/musical_expertise_copy*
T0*#
_output_shapes
:’’’’’’’’’

Dtransform/transform/scale_by_min_max_10/min_and_max/amax/PlaceholderPlaceholder*
shape: *
dtype0*
_output_shapes
: 

Ftransform/transform/scale_by_min_max_10/min_and_max/amax/Placeholder_1Placeholder*
dtype0*
_output_shapes
: *
shape: 

;transform/transform/scale_by_min_max_10/min_and_max/sub_1/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
y
4transform/transform/scale_by_min_max_10/Fill_1/valueConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
É
.transform/transform/scale_by_min_max_10/Fill_1Fill-transform/transform/scale_by_min_max_10/Shape4transform/transform/scale_by_min_max_10/Fill_1/value*
T0*#
_output_shapes
:’’’’’’’’’
r
-transform/transform/scale_by_min_max_10/mul/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
r
-transform/transform/scale_by_min_max_10/add/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
~
9transform/transform/scale_by_min_max_11/min_and_max/sub/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
7transform/transform/scale_by_min_max_11/min_and_max/subSub9transform/transform/scale_by_min_max_11/min_and_max/sub/x'transform/transform/inputs/nervous_copy*#
_output_shapes
:’’’’’’’’’*
T0

Dtransform/transform/scale_by_min_max_11/min_and_max/amax/PlaceholderPlaceholder*
dtype0*
_output_shapes
: *
shape: 

Ftransform/transform/scale_by_min_max_11/min_and_max/amax/Placeholder_1Placeholder*
dtype0*
_output_shapes
: *
shape: 

;transform/transform/scale_by_min_max_11/min_and_max/sub_1/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
y
4transform/transform/scale_by_min_max_11/Fill_1/valueConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
É
.transform/transform/scale_by_min_max_11/Fill_1Fill-transform/transform/scale_by_min_max_11/Shape4transform/transform/scale_by_min_max_11/Fill_1/value*
T0*#
_output_shapes
:’’’’’’’’’
r
-transform/transform/scale_by_min_max_11/mul/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
r
-transform/transform/scale_by_min_max_11/add/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
~
9transform/transform/scale_by_min_max_12/min_and_max/sub/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ń
7transform/transform/scale_by_min_max_12/min_and_max/subSub9transform/transform/scale_by_min_max_12/min_and_max/sub/x(transform/transform/inputs/outgoing_copy*
T0*#
_output_shapes
:’’’’’’’’’

Dtransform/transform/scale_by_min_max_12/min_and_max/amax/PlaceholderPlaceholder*
shape: *
dtype0*
_output_shapes
: 

Ftransform/transform/scale_by_min_max_12/min_and_max/amax/Placeholder_1Placeholder*
dtype0*
_output_shapes
: *
shape: 

;transform/transform/scale_by_min_max_12/min_and_max/sub_1/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
y
4transform/transform/scale_by_min_max_12/Fill_1/valueConst*
_output_shapes
: *
valueB
 *   ?*
dtype0
É
.transform/transform/scale_by_min_max_12/Fill_1Fill-transform/transform/scale_by_min_max_12/Shape4transform/transform/scale_by_min_max_12/Fill_1/value*#
_output_shapes
:’’’’’’’’’*
T0
r
-transform/transform/scale_by_min_max_12/mul/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
r
-transform/transform/scale_by_min_max_12/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
~
9transform/transform/scale_by_min_max_13/min_and_max/sub/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ó
7transform/transform/scale_by_min_max_13/min_and_max/subSub9transform/transform/scale_by_min_max_13/min_and_max/sub/x*transform/transform/inputs/positivity_copy*#
_output_shapes
:’’’’’’’’’*
T0

Dtransform/transform/scale_by_min_max_13/min_and_max/amax/PlaceholderPlaceholder*
dtype0*
_output_shapes
: *
shape: 

Ftransform/transform/scale_by_min_max_13/min_and_max/amax/Placeholder_1Placeholder*
_output_shapes
: *
shape: *
dtype0

;transform/transform/scale_by_min_max_13/min_and_max/sub_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *    
y
4transform/transform/scale_by_min_max_13/Fill_1/valueConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
É
.transform/transform/scale_by_min_max_13/Fill_1Fill-transform/transform/scale_by_min_max_13/Shape4transform/transform/scale_by_min_max_13/Fill_1/value*#
_output_shapes
:’’’’’’’’’*
T0
r
-transform/transform/scale_by_min_max_13/mul/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
r
-transform/transform/scale_by_min_max_13/add/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
~
9transform/transform/scale_by_min_max_14/min_and_max/sub/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ń
7transform/transform/scale_by_min_max_14/min_and_max/subSub9transform/transform/scale_by_min_max_14/min_and_max/sub/x(transform/transform/inputs/reserved_copy*
T0*#
_output_shapes
:’’’’’’’’’

Dtransform/transform/scale_by_min_max_14/min_and_max/amax/PlaceholderPlaceholder*
shape: *
dtype0*
_output_shapes
: 

Ftransform/transform/scale_by_min_max_14/min_and_max/amax/Placeholder_1Placeholder*
dtype0*
_output_shapes
: *
shape: 

;transform/transform/scale_by_min_max_14/min_and_max/sub_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *    
y
4transform/transform/scale_by_min_max_14/Fill_1/valueConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
É
.transform/transform/scale_by_min_max_14/Fill_1Fill-transform/transform/scale_by_min_max_14/Shape4transform/transform/scale_by_min_max_14/Fill_1/value*#
_output_shapes
:’’’’’’’’’*
T0
r
-transform/transform/scale_by_min_max_14/mul/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
r
-transform/transform/scale_by_min_max_14/add/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
~
9transform/transform/scale_by_min_max_15/min_and_max/sub/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ļ
7transform/transform/scale_by_min_max_15/min_and_max/subSub9transform/transform/scale_by_min_max_15/min_and_max/sub/x&transform/transform/inputs/stress_copy*
T0*#
_output_shapes
:’’’’’’’’’

Dtransform/transform/scale_by_min_max_15/min_and_max/amax/PlaceholderPlaceholder*
_output_shapes
: *
shape: *
dtype0

Ftransform/transform/scale_by_min_max_15/min_and_max/amax/Placeholder_1Placeholder*
dtype0*
_output_shapes
: *
shape: 

;transform/transform/scale_by_min_max_15/min_and_max/sub_1/xConst*
_output_shapes
: *
valueB
 *    *
dtype0
y
4transform/transform/scale_by_min_max_15/Fill_1/valueConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
É
.transform/transform/scale_by_min_max_15/Fill_1Fill-transform/transform/scale_by_min_max_15/Shape4transform/transform/scale_by_min_max_15/Fill_1/value*#
_output_shapes
:’’’’’’’’’*
T0
r
-transform/transform/scale_by_min_max_15/mul/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
r
-transform/transform/scale_by_min_max_15/add/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
~
9transform/transform/scale_by_min_max_16/min_and_max/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *    
Š
7transform/transform/scale_by_min_max_16/min_and_max/subSub9transform/transform/scale_by_min_max_16/min_and_max/sub/x'transform/transform/inputs/tension_copy*
T0*#
_output_shapes
:’’’’’’’’’

Dtransform/transform/scale_by_min_max_16/min_and_max/amax/PlaceholderPlaceholder*
dtype0*
_output_shapes
: *
shape: 

Ftransform/transform/scale_by_min_max_16/min_and_max/amax/Placeholder_1Placeholder*
_output_shapes
: *
shape: *
dtype0

;transform/transform/scale_by_min_max_16/min_and_max/sub_1/xConst*
_output_shapes
: *
valueB
 *    *
dtype0
y
4transform/transform/scale_by_min_max_16/Fill_1/valueConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
É
.transform/transform/scale_by_min_max_16/Fill_1Fill-transform/transform/scale_by_min_max_16/Shape4transform/transform/scale_by_min_max_16/Fill_1/value*#
_output_shapes
:’’’’’’’’’*
T0
r
-transform/transform/scale_by_min_max_16/mul/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
r
-transform/transform/scale_by_min_max_16/add/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
~
9transform/transform/scale_by_min_max_17/min_and_max/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ń
7transform/transform/scale_by_min_max_17/min_and_max/subSub9transform/transform/scale_by_min_max_17/min_and_max/sub/x(transform/transform/inputs/thorough_copy*
T0*#
_output_shapes
:’’’’’’’’’

Dtransform/transform/scale_by_min_max_17/min_and_max/amax/PlaceholderPlaceholder*
dtype0*
_output_shapes
: *
shape: 

Ftransform/transform/scale_by_min_max_17/min_and_max/amax/Placeholder_1Placeholder*
dtype0*
_output_shapes
: *
shape: 

;transform/transform/scale_by_min_max_17/min_and_max/sub_1/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
y
4transform/transform/scale_by_min_max_17/Fill_1/valueConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
É
.transform/transform/scale_by_min_max_17/Fill_1Fill-transform/transform/scale_by_min_max_17/Shape4transform/transform/scale_by_min_max_17/Fill_1/value*
T0*#
_output_shapes
:’’’’’’’’’
r
-transform/transform/scale_by_min_max_17/mul/yConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
r
-transform/transform/scale_by_min_max_17/add/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
~
9transform/transform/scale_by_min_max_18/min_and_max/sub/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ń
7transform/transform/scale_by_min_max_18/min_and_max/subSub9transform/transform/scale_by_min_max_18/min_and_max/sub/x(transform/transform/inputs/trusting_copy*#
_output_shapes
:’’’’’’’’’*
T0

Dtransform/transform/scale_by_min_max_18/min_and_max/amax/PlaceholderPlaceholder*
_output_shapes
: *
shape: *
dtype0

Ftransform/transform/scale_by_min_max_18/min_and_max/amax/Placeholder_1Placeholder*
dtype0*
_output_shapes
: *
shape: 

;transform/transform/scale_by_min_max_18/min_and_max/sub_1/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
y
4transform/transform/scale_by_min_max_18/Fill_1/valueConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
É
.transform/transform/scale_by_min_max_18/Fill_1Fill-transform/transform/scale_by_min_max_18/Shape4transform/transform/scale_by_min_max_18/Fill_1/value*#
_output_shapes
:’’’’’’’’’*
T0
r
-transform/transform/scale_by_min_max_18/mul/yConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
r
-transform/transform/scale_by_min_max_18/add/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
 
transform/transform/initNoOp
"
transform/transform/init_1NoOp
W
transform/Const_37Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
W
transform/Const_36Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
W
transform/Const_35Const*
valueB
 *  æ*
dtype0*
_output_shapes
: 
°
8transform/transform/scale_by_min_max_8/min_and_max/sub_1Sub:transform/transform/scale_by_min_max_8/min_and_max/sub_1/xtransform/Const_35*
_output_shapes
: *
T0
æ
*transform/transform/scale_by_min_max_8/subSub$transform/transform/inputs/lazy_copy8transform/transform/scale_by_min_max_8/min_and_max/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
W
transform/Const_34Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
W
transform/Const_33Const*
dtype0*
_output_shapes
: *
valueB
 *  æ
²
9transform/transform/scale_by_min_max_15/min_and_max/sub_1Sub;transform/transform/scale_by_min_max_15/min_and_max/sub_1/xtransform/Const_33*
T0*
_output_shapes
: 
Ć
+transform/transform/scale_by_min_max_15/subSub&transform/transform/inputs/stress_copy9transform/transform/scale_by_min_max_15/min_and_max/sub_1*#
_output_shapes
:’’’’’’’’’*
T0
W
transform/Const_32Const*
_output_shapes
: *
valueB
 *   @*
dtype0
W
transform/Const_31Const*
valueB
 *  æ*
dtype0*
_output_shapes
: 
°
8transform/transform/scale_by_min_max_4/min_and_max/sub_1Sub:transform/transform/scale_by_min_max_4/min_and_max/sub_1/xtransform/Const_31*
T0*
_output_shapes
: 
Å
*transform/transform/scale_by_min_max_4/subSub*transform/transform/inputs/engagement_copy8transform/transform/scale_by_min_max_4/min_and_max/sub_1*#
_output_shapes
:’’’’’’’’’*
T0
W
transform/Const_30Const*
valueB
 *  æ*
dtype0*
_output_shapes
: 
°
8transform/transform/scale_by_min_max_2/min_and_max/sub_1Sub:transform/transform/scale_by_min_max_2/min_and_max/sub_1/xtransform/Const_30*
T0*
_output_shapes
: 
Ć
*transform/transform/scale_by_min_max_2/subSub(transform/transform/inputs/artistic_copy8transform/transform/scale_by_min_max_2/min_and_max/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
¢
+transform/transform/scale_by_min_max_2/LessLess8transform/transform/scale_by_min_max_2/min_and_max/sub_1transform/Const_36*
T0*
_output_shapes
: 
¼
+transform/transform/scale_by_min_max_2/FillFill,transform/transform/scale_by_min_max_2/Shape+transform/transform/scale_by_min_max_2/Less*
T0
*#
_output_shapes
:’’’’’’’’’
¢
,transform/transform/scale_by_min_max_2/sub_1Subtransform/Const_368transform/transform/scale_by_min_max_2/min_and_max/sub_1*
T0*
_output_shapes
: 
Į
.transform/transform/scale_by_min_max_2/truedivRealDiv*transform/transform/scale_by_min_max_2/sub,transform/transform/scale_by_min_max_2/sub_1*#
_output_shapes
:’’’’’’’’’*
T0
ń
-transform/transform/scale_by_min_max_2/SelectSelect+transform/transform/scale_by_min_max_2/Fill.transform/transform/scale_by_min_max_2/truediv-transform/transform/scale_by_min_max_2/Fill_1*
T0*#
_output_shapes
:’’’’’’’’’
¼
*transform/transform/scale_by_min_max_2/mulMul-transform/transform/scale_by_min_max_2/Select,transform/transform/scale_by_min_max_2/mul/y*
T0*#
_output_shapes
:’’’’’’’’’
¹
*transform/transform/scale_by_min_max_2/addAdd*transform/transform/scale_by_min_max_2/mul,transform/transform/scale_by_min_max_2/add/y*#
_output_shapes
:’’’’’’’’’*
T0
W
transform/Const_29Const*
dtype0*
_output_shapes
: *
valueB
 *  æ
²
9transform/transform/scale_by_min_max_18/min_and_max/sub_1Sub;transform/transform/scale_by_min_max_18/min_and_max/sub_1/xtransform/Const_29*
_output_shapes
: *
T0
Å
+transform/transform/scale_by_min_max_18/subSub(transform/transform/inputs/trusting_copy9transform/transform/scale_by_min_max_18/min_and_max/sub_1*#
_output_shapes
:’’’’’’’’’*
T0
¤
,transform/transform/scale_by_min_max_18/LessLess9transform/transform/scale_by_min_max_18/min_and_max/sub_1transform/Const_34*
T0*
_output_shapes
: 
æ
,transform/transform/scale_by_min_max_18/FillFill-transform/transform/scale_by_min_max_18/Shape,transform/transform/scale_by_min_max_18/Less*#
_output_shapes
:’’’’’’’’’*
T0

¤
-transform/transform/scale_by_min_max_18/sub_1Subtransform/Const_349transform/transform/scale_by_min_max_18/min_and_max/sub_1*
_output_shapes
: *
T0
Ä
/transform/transform/scale_by_min_max_18/truedivRealDiv+transform/transform/scale_by_min_max_18/sub-transform/transform/scale_by_min_max_18/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
õ
.transform/transform/scale_by_min_max_18/SelectSelect,transform/transform/scale_by_min_max_18/Fill/transform/transform/scale_by_min_max_18/truediv.transform/transform/scale_by_min_max_18/Fill_1*#
_output_shapes
:’’’’’’’’’*
T0
æ
+transform/transform/scale_by_min_max_18/mulMul.transform/transform/scale_by_min_max_18/Select-transform/transform/scale_by_min_max_18/mul/y*
T0*#
_output_shapes
:’’’’’’’’’
¼
+transform/transform/scale_by_min_max_18/addAdd+transform/transform/scale_by_min_max_18/mul-transform/transform/scale_by_min_max_18/add/y*#
_output_shapes
:’’’’’’’’’*
T0
W
transform/Const_28Const*
valueB
 *  ņB*
dtype0*
_output_shapes
: 
W
transform/Const_27Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
W
transform/Const_26Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
W
transform/Const_25Const*
valueB
 *  æ*
dtype0*
_output_shapes
: 
²
9transform/transform/scale_by_min_max_17/min_and_max/sub_1Sub;transform/transform/scale_by_min_max_17/min_and_max/sub_1/xtransform/Const_25*
T0*
_output_shapes
: 
Å
+transform/transform/scale_by_min_max_17/subSub(transform/transform/inputs/thorough_copy9transform/transform/scale_by_min_max_17/min_and_max/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
W
transform/Const_24Const*
valueB
 *  æ*
dtype0*
_output_shapes
: 
°
8transform/transform/scale_by_min_max_9/min_and_max/sub_1Sub:transform/transform/scale_by_min_max_9/min_and_max/sub_1/xtransform/Const_24*
T0*
_output_shapes
: 
Ē
*transform/transform/scale_by_min_max_9/subSub,transform/transform/inputs/like_dislike_copy8transform/transform/scale_by_min_max_9/min_and_max/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
¢
+transform/transform/scale_by_min_max_9/LessLess8transform/transform/scale_by_min_max_9/min_and_max/sub_1transform/Const_26*
T0*
_output_shapes
: 
¼
+transform/transform/scale_by_min_max_9/FillFill,transform/transform/scale_by_min_max_9/Shape+transform/transform/scale_by_min_max_9/Less*
T0
*#
_output_shapes
:’’’’’’’’’
¢
,transform/transform/scale_by_min_max_9/sub_1Subtransform/Const_268transform/transform/scale_by_min_max_9/min_and_max/sub_1*
T0*
_output_shapes
: 
Į
.transform/transform/scale_by_min_max_9/truedivRealDiv*transform/transform/scale_by_min_max_9/sub,transform/transform/scale_by_min_max_9/sub_1*#
_output_shapes
:’’’’’’’’’*
T0
ń
-transform/transform/scale_by_min_max_9/SelectSelect+transform/transform/scale_by_min_max_9/Fill.transform/transform/scale_by_min_max_9/truediv-transform/transform/scale_by_min_max_9/Fill_1*
T0*#
_output_shapes
:’’’’’’’’’
¼
*transform/transform/scale_by_min_max_9/mulMul-transform/transform/scale_by_min_max_9/Select,transform/transform/scale_by_min_max_9/mul/y*
T0*#
_output_shapes
:’’’’’’’’’
¹
*transform/transform/scale_by_min_max_9/addAdd*transform/transform/scale_by_min_max_9/mul,transform/transform/scale_by_min_max_9/add/y*#
_output_shapes
:’’’’’’’’’*
T0
W
transform/Const_23Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
W
transform/Const_22Const*
dtype0*
_output_shapes
: *
valueB
 *  æ
²
9transform/transform/scale_by_min_max_16/min_and_max/sub_1Sub;transform/transform/scale_by_min_max_16/min_and_max/sub_1/xtransform/Const_22*
T0*
_output_shapes
: 
Ä
+transform/transform/scale_by_min_max_16/subSub'transform/transform/inputs/tension_copy9transform/transform/scale_by_min_max_16/min_and_max/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
W
transform/Const_21Const*
_output_shapes
: *
valueB
 *   @*
dtype0
W
transform/Const_20Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
W
transform/Const_19Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
W
transform/Const_18Const*
_output_shapes
: *
valueB
 *   @*
dtype0
¢
+transform/transform/scale_by_min_max_4/LessLess8transform/transform/scale_by_min_max_4/min_and_max/sub_1transform/Const_18*
_output_shapes
: *
T0
¼
+transform/transform/scale_by_min_max_4/FillFill,transform/transform/scale_by_min_max_4/Shape+transform/transform/scale_by_min_max_4/Less*#
_output_shapes
:’’’’’’’’’*
T0

¢
,transform/transform/scale_by_min_max_4/sub_1Subtransform/Const_188transform/transform/scale_by_min_max_4/min_and_max/sub_1*
_output_shapes
: *
T0
Į
.transform/transform/scale_by_min_max_4/truedivRealDiv*transform/transform/scale_by_min_max_4/sub,transform/transform/scale_by_min_max_4/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
ń
-transform/transform/scale_by_min_max_4/SelectSelect+transform/transform/scale_by_min_max_4/Fill.transform/transform/scale_by_min_max_4/truediv-transform/transform/scale_by_min_max_4/Fill_1*
T0*#
_output_shapes
:’’’’’’’’’
¼
*transform/transform/scale_by_min_max_4/mulMul-transform/transform/scale_by_min_max_4/Select,transform/transform/scale_by_min_max_4/mul/y*
T0*#
_output_shapes
:’’’’’’’’’
¹
*transform/transform/scale_by_min_max_4/addAdd*transform/transform/scale_by_min_max_4/mul,transform/transform/scale_by_min_max_4/add/y*
T0*#
_output_shapes
:’’’’’’’’’
W
transform/Const_17Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
W
transform/Const_16Const*
dtype0*
_output_shapes
: *
valueB
 *  æ
°
8transform/transform/scale_by_min_max_6/min_and_max/sub_1Sub:transform/transform/scale_by_min_max_6/min_and_max/sub_1/xtransform/Const_16*
T0*
_output_shapes
: 
Ą
*transform/transform/scale_by_min_max_6/subSub%transform/transform/inputs/fault_copy8transform/transform/scale_by_min_max_6/min_and_max/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
¢
+transform/transform/scale_by_min_max_6/LessLess8transform/transform/scale_by_min_max_6/min_and_max/sub_1transform/Const_19*
T0*
_output_shapes
: 
¼
+transform/transform/scale_by_min_max_6/FillFill,transform/transform/scale_by_min_max_6/Shape+transform/transform/scale_by_min_max_6/Less*#
_output_shapes
:’’’’’’’’’*
T0

¢
,transform/transform/scale_by_min_max_6/sub_1Subtransform/Const_198transform/transform/scale_by_min_max_6/min_and_max/sub_1*
_output_shapes
: *
T0
Į
.transform/transform/scale_by_min_max_6/truedivRealDiv*transform/transform/scale_by_min_max_6/sub,transform/transform/scale_by_min_max_6/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
ń
-transform/transform/scale_by_min_max_6/SelectSelect+transform/transform/scale_by_min_max_6/Fill.transform/transform/scale_by_min_max_6/truediv-transform/transform/scale_by_min_max_6/Fill_1*#
_output_shapes
:’’’’’’’’’*
T0
¼
*transform/transform/scale_by_min_max_6/mulMul-transform/transform/scale_by_min_max_6/Select,transform/transform/scale_by_min_max_6/mul/y*
T0*#
_output_shapes
:’’’’’’’’’
¹
*transform/transform/scale_by_min_max_6/addAdd*transform/transform/scale_by_min_max_6/mul,transform/transform/scale_by_min_max_6/add/y*
T0*#
_output_shapes
:’’’’’’’’’
W
transform/Const_15Const*
_output_shapes
: *
valueB
 *  æ*
dtype0
²
9transform/transform/scale_by_min_max_12/min_and_max/sub_1Sub;transform/transform/scale_by_min_max_12/min_and_max/sub_1/xtransform/Const_15*
T0*
_output_shapes
: 
Å
+transform/transform/scale_by_min_max_12/subSub(transform/transform/inputs/outgoing_copy9transform/transform/scale_by_min_max_12/min_and_max/sub_1*#
_output_shapes
:’’’’’’’’’*
T0
W
transform/Const_14Const*
_output_shapes
: *
valueB
 *  æ*
dtype0
²
9transform/transform/scale_by_min_max_14/min_and_max/sub_1Sub;transform/transform/scale_by_min_max_14/min_and_max/sub_1/xtransform/Const_14*
T0*
_output_shapes
: 
Å
+transform/transform/scale_by_min_max_14/subSub(transform/transform/inputs/reserved_copy9transform/transform/scale_by_min_max_14/min_and_max/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
¤
,transform/transform/scale_by_min_max_14/LessLess9transform/transform/scale_by_min_max_14/min_and_max/sub_1transform/Const_37*
T0*
_output_shapes
: 
æ
,transform/transform/scale_by_min_max_14/FillFill-transform/transform/scale_by_min_max_14/Shape,transform/transform/scale_by_min_max_14/Less*#
_output_shapes
:’’’’’’’’’*
T0

¤
-transform/transform/scale_by_min_max_14/sub_1Subtransform/Const_379transform/transform/scale_by_min_max_14/min_and_max/sub_1*
T0*
_output_shapes
: 
Ä
/transform/transform/scale_by_min_max_14/truedivRealDiv+transform/transform/scale_by_min_max_14/sub-transform/transform/scale_by_min_max_14/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
õ
.transform/transform/scale_by_min_max_14/SelectSelect,transform/transform/scale_by_min_max_14/Fill/transform/transform/scale_by_min_max_14/truediv.transform/transform/scale_by_min_max_14/Fill_1*
T0*#
_output_shapes
:’’’’’’’’’
æ
+transform/transform/scale_by_min_max_14/mulMul.transform/transform/scale_by_min_max_14/Select-transform/transform/scale_by_min_max_14/mul/y*
T0*#
_output_shapes
:’’’’’’’’’
¼
+transform/transform/scale_by_min_max_14/addAdd+transform/transform/scale_by_min_max_14/mul-transform/transform/scale_by_min_max_14/add/y*
T0*#
_output_shapes
:’’’’’’’’’
W
transform/Const_13Const*
valueB
 *  æ*
dtype0*
_output_shapes
: 
²
9transform/transform/scale_by_min_max_10/min_and_max/sub_1Sub;transform/transform/scale_by_min_max_10/min_and_max/sub_1/xtransform/Const_13*
T0*
_output_shapes
: 
Ī
+transform/transform/scale_by_min_max_10/subSub1transform/transform/inputs/musical_expertise_copy9transform/transform/scale_by_min_max_10/min_and_max/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
¤
,transform/transform/scale_by_min_max_10/LessLess9transform/transform/scale_by_min_max_10/min_and_max/sub_1transform/Const_23*
T0*
_output_shapes
: 
æ
,transform/transform/scale_by_min_max_10/FillFill-transform/transform/scale_by_min_max_10/Shape,transform/transform/scale_by_min_max_10/Less*
T0
*#
_output_shapes
:’’’’’’’’’
¤
-transform/transform/scale_by_min_max_10/sub_1Subtransform/Const_239transform/transform/scale_by_min_max_10/min_and_max/sub_1*
T0*
_output_shapes
: 
Ä
/transform/transform/scale_by_min_max_10/truedivRealDiv+transform/transform/scale_by_min_max_10/sub-transform/transform/scale_by_min_max_10/sub_1*#
_output_shapes
:’’’’’’’’’*
T0
õ
.transform/transform/scale_by_min_max_10/SelectSelect,transform/transform/scale_by_min_max_10/Fill/transform/transform/scale_by_min_max_10/truediv.transform/transform/scale_by_min_max_10/Fill_1*
T0*#
_output_shapes
:’’’’’’’’’
æ
+transform/transform/scale_by_min_max_10/mulMul.transform/transform/scale_by_min_max_10/Select-transform/transform/scale_by_min_max_10/mul/y*
T0*#
_output_shapes
:’’’’’’’’’
¼
+transform/transform/scale_by_min_max_10/addAdd+transform/transform/scale_by_min_max_10/mul-transform/transform/scale_by_min_max_10/add/y*
T0*#
_output_shapes
:’’’’’’’’’
W
transform/Const_12Const*
dtype0*
_output_shapes
: *
valueB
 *  æ
²
9transform/transform/scale_by_min_max_11/min_and_max/sub_1Sub;transform/transform/scale_by_min_max_11/min_and_max/sub_1/xtransform/Const_12*
T0*
_output_shapes
: 
Ä
+transform/transform/scale_by_min_max_11/subSub'transform/transform/inputs/nervous_copy9transform/transform/scale_by_min_max_11/min_and_max/sub_1*#
_output_shapes
:’’’’’’’’’*
T0
¤
,transform/transform/scale_by_min_max_11/LessLess9transform/transform/scale_by_min_max_11/min_and_max/sub_1transform/Const_32*
T0*
_output_shapes
: 
æ
,transform/transform/scale_by_min_max_11/FillFill-transform/transform/scale_by_min_max_11/Shape,transform/transform/scale_by_min_max_11/Less*
T0
*#
_output_shapes
:’’’’’’’’’
¤
-transform/transform/scale_by_min_max_11/sub_1Subtransform/Const_329transform/transform/scale_by_min_max_11/min_and_max/sub_1*
_output_shapes
: *
T0
Ä
/transform/transform/scale_by_min_max_11/truedivRealDiv+transform/transform/scale_by_min_max_11/sub-transform/transform/scale_by_min_max_11/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
õ
.transform/transform/scale_by_min_max_11/SelectSelect,transform/transform/scale_by_min_max_11/Fill/transform/transform/scale_by_min_max_11/truediv.transform/transform/scale_by_min_max_11/Fill_1*
T0*#
_output_shapes
:’’’’’’’’’
æ
+transform/transform/scale_by_min_max_11/mulMul.transform/transform/scale_by_min_max_11/Select-transform/transform/scale_by_min_max_11/mul/y*#
_output_shapes
:’’’’’’’’’*
T0
¼
+transform/transform/scale_by_min_max_11/addAdd+transform/transform/scale_by_min_max_11/mul-transform/transform/scale_by_min_max_11/add/y*
T0*#
_output_shapes
:’’’’’’’’’
W
transform/Const_11Const*
valueB
 *  æ*
dtype0*
_output_shapes
: 
°
8transform/transform/scale_by_min_max_7/min_and_max/sub_1Sub:transform/transform/scale_by_min_max_7/min_and_max/sub_1/xtransform/Const_11*
T0*
_output_shapes
: 
Ę
*transform/transform/scale_by_min_max_7/subSub+transform/transform/inputs/imagination_copy8transform/transform/scale_by_min_max_7/min_and_max/sub_1*#
_output_shapes
:’’’’’’’’’*
T0
¢
+transform/transform/scale_by_min_max_7/LessLess8transform/transform/scale_by_min_max_7/min_and_max/sub_1transform/Const_20*
_output_shapes
: *
T0
¼
+transform/transform/scale_by_min_max_7/FillFill,transform/transform/scale_by_min_max_7/Shape+transform/transform/scale_by_min_max_7/Less*#
_output_shapes
:’’’’’’’’’*
T0

¢
,transform/transform/scale_by_min_max_7/sub_1Subtransform/Const_208transform/transform/scale_by_min_max_7/min_and_max/sub_1*
T0*
_output_shapes
: 
Į
.transform/transform/scale_by_min_max_7/truedivRealDiv*transform/transform/scale_by_min_max_7/sub,transform/transform/scale_by_min_max_7/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
ń
-transform/transform/scale_by_min_max_7/SelectSelect+transform/transform/scale_by_min_max_7/Fill.transform/transform/scale_by_min_max_7/truediv-transform/transform/scale_by_min_max_7/Fill_1*
T0*#
_output_shapes
:’’’’’’’’’
¼
*transform/transform/scale_by_min_max_7/mulMul-transform/transform/scale_by_min_max_7/Select,transform/transform/scale_by_min_max_7/mul/y*
T0*#
_output_shapes
:’’’’’’’’’
¹
*transform/transform/scale_by_min_max_7/addAdd*transform/transform/scale_by_min_max_7/mul,transform/transform/scale_by_min_max_7/add/y*#
_output_shapes
:’’’’’’’’’*
T0
W
transform/Const_10Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
¤
,transform/transform/scale_by_min_max_12/LessLess9transform/transform/scale_by_min_max_12/min_and_max/sub_1transform/Const_10*
_output_shapes
: *
T0
æ
,transform/transform/scale_by_min_max_12/FillFill-transform/transform/scale_by_min_max_12/Shape,transform/transform/scale_by_min_max_12/Less*
T0
*#
_output_shapes
:’’’’’’’’’
¤
-transform/transform/scale_by_min_max_12/sub_1Subtransform/Const_109transform/transform/scale_by_min_max_12/min_and_max/sub_1*
T0*
_output_shapes
: 
Ä
/transform/transform/scale_by_min_max_12/truedivRealDiv+transform/transform/scale_by_min_max_12/sub-transform/transform/scale_by_min_max_12/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
õ
.transform/transform/scale_by_min_max_12/SelectSelect,transform/transform/scale_by_min_max_12/Fill/transform/transform/scale_by_min_max_12/truediv.transform/transform/scale_by_min_max_12/Fill_1*
T0*#
_output_shapes
:’’’’’’’’’
æ
+transform/transform/scale_by_min_max_12/mulMul.transform/transform/scale_by_min_max_12/Select-transform/transform/scale_by_min_max_12/mul/y*
T0*#
_output_shapes
:’’’’’’’’’
¼
+transform/transform/scale_by_min_max_12/addAdd+transform/transform/scale_by_min_max_12/mul-transform/transform/scale_by_min_max_12/add/y*
T0*#
_output_shapes
:’’’’’’’’’
V
transform/Const_9Const*
valueB
 *  æ*
dtype0*
_output_shapes
: 
«
6transform/transform/scale_by_min_max/min_and_max/sub_1Sub8transform/transform/scale_by_min_max/min_and_max/sub_1/xtransform/Const_9*
T0*
_output_shapes
: 
æ
(transform/transform/scale_by_min_max/subSub(transform/transform/inputs/activity_copy6transform/transform/scale_by_min_max/min_and_max/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
V
transform/Const_8Const*
valueB
 *   @*
dtype0*
_output_shapes
: 

)transform/transform/scale_by_min_max/LessLess6transform/transform/scale_by_min_max/min_and_max/sub_1transform/Const_8*
T0*
_output_shapes
: 
¶
)transform/transform/scale_by_min_max/FillFill*transform/transform/scale_by_min_max/Shape)transform/transform/scale_by_min_max/Less*
T0
*#
_output_shapes
:’’’’’’’’’

*transform/transform/scale_by_min_max/sub_1Subtransform/Const_86transform/transform/scale_by_min_max/min_and_max/sub_1*
T0*
_output_shapes
: 
»
,transform/transform/scale_by_min_max/truedivRealDiv(transform/transform/scale_by_min_max/sub*transform/transform/scale_by_min_max/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
é
+transform/transform/scale_by_min_max/SelectSelect)transform/transform/scale_by_min_max/Fill,transform/transform/scale_by_min_max/truediv+transform/transform/scale_by_min_max/Fill_1*#
_output_shapes
:’’’’’’’’’*
T0
¶
(transform/transform/scale_by_min_max/mulMul+transform/transform/scale_by_min_max/Select*transform/transform/scale_by_min_max/mul/y*#
_output_shapes
:’’’’’’’’’*
T0
³
(transform/transform/scale_by_min_max/addAdd(transform/transform/scale_by_min_max/mul*transform/transform/scale_by_min_max/add/y*#
_output_shapes
:’’’’’’’’’*
T0
V
transform/Const_7Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
£
,transform/transform/scale_by_min_max_15/LessLess9transform/transform/scale_by_min_max_15/min_and_max/sub_1transform/Const_7*
T0*
_output_shapes
: 
æ
,transform/transform/scale_by_min_max_15/FillFill-transform/transform/scale_by_min_max_15/Shape,transform/transform/scale_by_min_max_15/Less*
T0
*#
_output_shapes
:’’’’’’’’’
£
-transform/transform/scale_by_min_max_15/sub_1Subtransform/Const_79transform/transform/scale_by_min_max_15/min_and_max/sub_1*
_output_shapes
: *
T0
Ä
/transform/transform/scale_by_min_max_15/truedivRealDiv+transform/transform/scale_by_min_max_15/sub-transform/transform/scale_by_min_max_15/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
õ
.transform/transform/scale_by_min_max_15/SelectSelect,transform/transform/scale_by_min_max_15/Fill/transform/transform/scale_by_min_max_15/truediv.transform/transform/scale_by_min_max_15/Fill_1*#
_output_shapes
:’’’’’’’’’*
T0
æ
+transform/transform/scale_by_min_max_15/mulMul.transform/transform/scale_by_min_max_15/Select-transform/transform/scale_by_min_max_15/mul/y*#
_output_shapes
:’’’’’’’’’*
T0
¼
+transform/transform/scale_by_min_max_15/addAdd+transform/transform/scale_by_min_max_15/mul-transform/transform/scale_by_min_max_15/add/y*
T0*#
_output_shapes
:’’’’’’’’’
V
transform/Const_6Const*
_output_shapes
: *
valueB
 *   @*
dtype0
£
,transform/transform/scale_by_min_max_17/LessLess9transform/transform/scale_by_min_max_17/min_and_max/sub_1transform/Const_6*
_output_shapes
: *
T0
æ
,transform/transform/scale_by_min_max_17/FillFill-transform/transform/scale_by_min_max_17/Shape,transform/transform/scale_by_min_max_17/Less*
T0
*#
_output_shapes
:’’’’’’’’’
£
-transform/transform/scale_by_min_max_17/sub_1Subtransform/Const_69transform/transform/scale_by_min_max_17/min_and_max/sub_1*
_output_shapes
: *
T0
Ä
/transform/transform/scale_by_min_max_17/truedivRealDiv+transform/transform/scale_by_min_max_17/sub-transform/transform/scale_by_min_max_17/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
õ
.transform/transform/scale_by_min_max_17/SelectSelect,transform/transform/scale_by_min_max_17/Fill/transform/transform/scale_by_min_max_17/truediv.transform/transform/scale_by_min_max_17/Fill_1*#
_output_shapes
:’’’’’’’’’*
T0
æ
+transform/transform/scale_by_min_max_17/mulMul.transform/transform/scale_by_min_max_17/Select-transform/transform/scale_by_min_max_17/mul/y*
T0*#
_output_shapes
:’’’’’’’’’
¼
+transform/transform/scale_by_min_max_17/addAdd+transform/transform/scale_by_min_max_17/mul-transform/transform/scale_by_min_max_17/add/y*#
_output_shapes
:’’’’’’’’’*
T0
V
transform/Const_5Const*
_output_shapes
: *
valueB
 *   @*
dtype0
£
,transform/transform/scale_by_min_max_16/LessLess9transform/transform/scale_by_min_max_16/min_and_max/sub_1transform/Const_5*
T0*
_output_shapes
: 
æ
,transform/transform/scale_by_min_max_16/FillFill-transform/transform/scale_by_min_max_16/Shape,transform/transform/scale_by_min_max_16/Less*
T0
*#
_output_shapes
:’’’’’’’’’
£
-transform/transform/scale_by_min_max_16/sub_1Subtransform/Const_59transform/transform/scale_by_min_max_16/min_and_max/sub_1*
_output_shapes
: *
T0
Ä
/transform/transform/scale_by_min_max_16/truedivRealDiv+transform/transform/scale_by_min_max_16/sub-transform/transform/scale_by_min_max_16/sub_1*#
_output_shapes
:’’’’’’’’’*
T0
õ
.transform/transform/scale_by_min_max_16/SelectSelect,transform/transform/scale_by_min_max_16/Fill/transform/transform/scale_by_min_max_16/truediv.transform/transform/scale_by_min_max_16/Fill_1*
T0*#
_output_shapes
:’’’’’’’’’
æ
+transform/transform/scale_by_min_max_16/mulMul.transform/transform/scale_by_min_max_16/Select-transform/transform/scale_by_min_max_16/mul/y*
T0*#
_output_shapes
:’’’’’’’’’
¼
+transform/transform/scale_by_min_max_16/addAdd+transform/transform/scale_by_min_max_16/mul-transform/transform/scale_by_min_max_16/add/y*
T0*#
_output_shapes
:’’’’’’’’’
V
transform/Const_4Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
”
+transform/transform/scale_by_min_max_8/LessLess8transform/transform/scale_by_min_max_8/min_and_max/sub_1transform/Const_4*
T0*
_output_shapes
: 
¼
+transform/transform/scale_by_min_max_8/FillFill,transform/transform/scale_by_min_max_8/Shape+transform/transform/scale_by_min_max_8/Less*#
_output_shapes
:’’’’’’’’’*
T0

”
,transform/transform/scale_by_min_max_8/sub_1Subtransform/Const_48transform/transform/scale_by_min_max_8/min_and_max/sub_1*
_output_shapes
: *
T0
Į
.transform/transform/scale_by_min_max_8/truedivRealDiv*transform/transform/scale_by_min_max_8/sub,transform/transform/scale_by_min_max_8/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
ń
-transform/transform/scale_by_min_max_8/SelectSelect+transform/transform/scale_by_min_max_8/Fill.transform/transform/scale_by_min_max_8/truediv-transform/transform/scale_by_min_max_8/Fill_1*#
_output_shapes
:’’’’’’’’’*
T0
¼
*transform/transform/scale_by_min_max_8/mulMul-transform/transform/scale_by_min_max_8/Select,transform/transform/scale_by_min_max_8/mul/y*#
_output_shapes
:’’’’’’’’’*
T0
¹
*transform/transform/scale_by_min_max_8/addAdd*transform/transform/scale_by_min_max_8/mul,transform/transform/scale_by_min_max_8/add/y*#
_output_shapes
:’’’’’’’’’*
T0
V
transform/Const_3Const*
_output_shapes
: *
valueB
 *  æ*
dtype0
±
9transform/transform/scale_by_min_max_13/min_and_max/sub_1Sub;transform/transform/scale_by_min_max_13/min_and_max/sub_1/xtransform/Const_3*
T0*
_output_shapes
: 
Ē
+transform/transform/scale_by_min_max_13/subSub*transform/transform/inputs/positivity_copy9transform/transform/scale_by_min_max_13/min_and_max/sub_1*#
_output_shapes
:’’’’’’’’’*
T0
¤
,transform/transform/scale_by_min_max_13/LessLess9transform/transform/scale_by_min_max_13/min_and_max/sub_1transform/Const_27*
_output_shapes
: *
T0
æ
,transform/transform/scale_by_min_max_13/FillFill-transform/transform/scale_by_min_max_13/Shape,transform/transform/scale_by_min_max_13/Less*
T0
*#
_output_shapes
:’’’’’’’’’
¤
-transform/transform/scale_by_min_max_13/sub_1Subtransform/Const_279transform/transform/scale_by_min_max_13/min_and_max/sub_1*
_output_shapes
: *
T0
Ä
/transform/transform/scale_by_min_max_13/truedivRealDiv+transform/transform/scale_by_min_max_13/sub-transform/transform/scale_by_min_max_13/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
õ
.transform/transform/scale_by_min_max_13/SelectSelect,transform/transform/scale_by_min_max_13/Fill/transform/transform/scale_by_min_max_13/truediv.transform/transform/scale_by_min_max_13/Fill_1*
T0*#
_output_shapes
:’’’’’’’’’
æ
+transform/transform/scale_by_min_max_13/mulMul.transform/transform/scale_by_min_max_13/Select-transform/transform/scale_by_min_max_13/mul/y*
T0*#
_output_shapes
:’’’’’’’’’
¼
+transform/transform/scale_by_min_max_13/addAdd+transform/transform/scale_by_min_max_13/mul-transform/transform/scale_by_min_max_13/add/y*
T0*#
_output_shapes
:’’’’’’’’’
V
transform/Const_2Const*
valueB
 *  æ*
dtype0*
_output_shapes
: 
Æ
8transform/transform/scale_by_min_max_5/min_and_max/sub_1Sub:transform/transform/scale_by_min_max_5/min_and_max/sub_1/xtransform/Const_2*
T0*
_output_shapes
: 
Ę
*transform/transform/scale_by_min_max_5/subSub+transform/transform/inputs/familiarity_copy8transform/transform/scale_by_min_max_5/min_and_max/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
¢
+transform/transform/scale_by_min_max_5/LessLess8transform/transform/scale_by_min_max_5/min_and_max/sub_1transform/Const_17*
T0*
_output_shapes
: 
¼
+transform/transform/scale_by_min_max_5/FillFill,transform/transform/scale_by_min_max_5/Shape+transform/transform/scale_by_min_max_5/Less*#
_output_shapes
:’’’’’’’’’*
T0

¢
,transform/transform/scale_by_min_max_5/sub_1Subtransform/Const_178transform/transform/scale_by_min_max_5/min_and_max/sub_1*
T0*
_output_shapes
: 
Į
.transform/transform/scale_by_min_max_5/truedivRealDiv*transform/transform/scale_by_min_max_5/sub,transform/transform/scale_by_min_max_5/sub_1*#
_output_shapes
:’’’’’’’’’*
T0
ń
-transform/transform/scale_by_min_max_5/SelectSelect+transform/transform/scale_by_min_max_5/Fill.transform/transform/scale_by_min_max_5/truediv-transform/transform/scale_by_min_max_5/Fill_1*#
_output_shapes
:’’’’’’’’’*
T0
¼
*transform/transform/scale_by_min_max_5/mulMul-transform/transform/scale_by_min_max_5/Select,transform/transform/scale_by_min_max_5/mul/y*#
_output_shapes
:’’’’’’’’’*
T0
¹
*transform/transform/scale_by_min_max_5/addAdd*transform/transform/scale_by_min_max_5/mul,transform/transform/scale_by_min_max_5/add/y*
T0*#
_output_shapes
:’’’’’’’’’
V
transform/Const_1Const*
valueB
 *  æ*
dtype0*
_output_shapes
: 
Æ
8transform/transform/scale_by_min_max_3/min_and_max/sub_1Sub:transform/transform/scale_by_min_max_3/min_and_max/sub_1/xtransform/Const_1*
T0*
_output_shapes
: 
Č
*transform/transform/scale_by_min_max_3/subSub-transform/transform/inputs/concentration_copy8transform/transform/scale_by_min_max_3/min_and_max/sub_1*
T0*#
_output_shapes
:’’’’’’’’’
¢
+transform/transform/scale_by_min_max_3/LessLess8transform/transform/scale_by_min_max_3/min_and_max/sub_1transform/Const_21*
T0*
_output_shapes
: 
¼
+transform/transform/scale_by_min_max_3/FillFill,transform/transform/scale_by_min_max_3/Shape+transform/transform/scale_by_min_max_3/Less*
T0
*#
_output_shapes
:’’’’’’’’’
¢
,transform/transform/scale_by_min_max_3/sub_1Subtransform/Const_218transform/transform/scale_by_min_max_3/min_and_max/sub_1*
T0*
_output_shapes
: 
Į
.transform/transform/scale_by_min_max_3/truedivRealDiv*transform/transform/scale_by_min_max_3/sub,transform/transform/scale_by_min_max_3/sub_1*#
_output_shapes
:’’’’’’’’’*
T0
ń
-transform/transform/scale_by_min_max_3/SelectSelect+transform/transform/scale_by_min_max_3/Fill.transform/transform/scale_by_min_max_3/truediv-transform/transform/scale_by_min_max_3/Fill_1*
T0*#
_output_shapes
:’’’’’’’’’
¼
*transform/transform/scale_by_min_max_3/mulMul-transform/transform/scale_by_min_max_3/Select,transform/transform/scale_by_min_max_3/mul/y*
T0*#
_output_shapes
:’’’’’’’’’
¹
*transform/transform/scale_by_min_max_3/addAdd*transform/transform/scale_by_min_max_3/mul,transform/transform/scale_by_min_max_3/add/y*#
_output_shapes
:’’’’’’’’’*
T0
T
transform/ConstConst*
valueB
 *  æ*
dtype0*
_output_shapes
: 
­
8transform/transform/scale_by_min_max_1/min_and_max/sub_1Sub:transform/transform/scale_by_min_max_1/min_and_max/sub_1/xtransform/Const*
_output_shapes
: *
T0
¾
*transform/transform/scale_by_min_max_1/subSub#transform/transform/inputs/age_copy8transform/transform/scale_by_min_max_1/min_and_max/sub_1*#
_output_shapes
:’’’’’’’’’*
T0
¢
+transform/transform/scale_by_min_max_1/LessLess8transform/transform/scale_by_min_max_1/min_and_max/sub_1transform/Const_28*
_output_shapes
: *
T0
¼
+transform/transform/scale_by_min_max_1/FillFill,transform/transform/scale_by_min_max_1/Shape+transform/transform/scale_by_min_max_1/Less*
T0
*#
_output_shapes
:’’’’’’’’’
¢
,transform/transform/scale_by_min_max_1/sub_1Subtransform/Const_288transform/transform/scale_by_min_max_1/min_and_max/sub_1*
T0*
_output_shapes
: 
Į
.transform/transform/scale_by_min_max_1/truedivRealDiv*transform/transform/scale_by_min_max_1/sub,transform/transform/scale_by_min_max_1/sub_1*#
_output_shapes
:’’’’’’’’’*
T0
ń
-transform/transform/scale_by_min_max_1/SelectSelect+transform/transform/scale_by_min_max_1/Fill.transform/transform/scale_by_min_max_1/truediv-transform/transform/scale_by_min_max_1/Fill_1*#
_output_shapes
:’’’’’’’’’*
T0
¼
*transform/transform/scale_by_min_max_1/mulMul-transform/transform/scale_by_min_max_1/Select,transform/transform/scale_by_min_max_1/mul/y*
T0*#
_output_shapes
:’’’’’’’’’
¹
*transform/transform/scale_by_min_max_1/addAdd*transform/transform/scale_by_min_max_1/mul,transform/transform/scale_by_min_max_1/add/y*
T0*#
_output_shapes
:’’’’’’’’’
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
l
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:* 
valueBBglobal_step
e
save/SaveV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
w
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_step*
dtypes
2	
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_class
loc:@save/Const*
_output_shapes
: *
T0
~
save/RestoreV2/tensor_namesConst"/device:CPU:0* 
valueBBglobal_step*
dtype0*
_output_shapes
:
w
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2	
s
save/AssignAssignglobal_stepsave/RestoreV2*
T0	*
_class
loc:@global_step*
_output_shapes
: 
&
save/restore_allNoOp^save/Assign

Idnn/input_from_feature_columns/input_layer/music_pref_none/ExpandDims/dimConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 

Ednn/input_from_feature_columns/input_layer/music_pref_none/ExpandDims
ExpandDims/transform/transform/inputs/music_pref_none_copyIdnn/input_from_feature_columns/input_layer/music_pref_none/ExpandDims/dim*'
_output_shapes
:’’’’’’’’’*
T0	
Ņ
Bdnn/input_from_feature_columns/input_layer/music_pref_none/ToFloatCastEdnn/input_from_feature_columns/input_layer/music_pref_none/ExpandDims*

SrcT0	*

DstT0*'
_output_shapes
:’’’’’’’’’
²
@dnn/input_from_feature_columns/input_layer/music_pref_none/ShapeShapeBdnn/input_from_feature_columns/input_layer/music_pref_none/ToFloat*
T0*
_output_shapes
:

Ndnn/input_from_feature_columns/input_layer/music_pref_none/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Pdnn/input_from_feature_columns/input_layer/music_pref_none/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Pdnn/input_from_feature_columns/input_layer/music_pref_none/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ō
Hdnn/input_from_feature_columns/input_layer/music_pref_none/strided_sliceStridedSlice@dnn/input_from_feature_columns/input_layer/music_pref_none/ShapeNdnn/input_from_feature_columns/input_layer/music_pref_none/strided_slice/stackPdnn/input_from_feature_columns/input_layer/music_pref_none/strided_slice/stack_1Pdnn/input_from_feature_columns/input_layer/music_pref_none/strided_slice/stack_2*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask

Jdnn/input_from_feature_columns/input_layer/music_pref_none/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :

Hdnn/input_from_feature_columns/input_layer/music_pref_none/Reshape/shapePackHdnn/input_from_feature_columns/input_layer/music_pref_none/strided_sliceJdnn/input_from_feature_columns/input_layer/music_pref_none/Reshape/shape/1*
T0*
N*
_output_shapes
:

Bdnn/input_from_feature_columns/input_layer/music_pref_none/ReshapeReshapeBdnn/input_from_feature_columns/input_layer/music_pref_none/ToFloatHdnn/input_from_feature_columns/input_layer/music_pref_none/Reshape/shape*'
_output_shapes
:’’’’’’’’’*
T0

Odnn/input_from_feature_columns/input_layer/nationality_indicator/ExpandDims/dimConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 
ü
Kdnn/input_from_feature_columns/input_layer/nationality_indicator/ExpandDims
ExpandDimstransform/transform/Identity_2Odnn/input_from_feature_columns/input_layer/nationality_indicator/ExpandDims/dim*
T0*'
_output_shapes
:’’’’’’’’’
 
_dnn/input_from_feature_columns/input_layer/nationality_indicator/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
Å
Ydnn/input_from_feature_columns/input_layer/nationality_indicator/to_sparse_input/NotEqualNotEqualKdnn/input_from_feature_columns/input_layer/nationality_indicator/ExpandDims_dnn/input_from_feature_columns/input_layer/nationality_indicator/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:’’’’’’’’’
å
Xdnn/input_from_feature_columns/input_layer/nationality_indicator/to_sparse_input/indicesWhereYdnn/input_from_feature_columns/input_layer/nationality_indicator/to_sparse_input/NotEqual*'
_output_shapes
:’’’’’’’’’
Ī
Wdnn/input_from_feature_columns/input_layer/nationality_indicator/to_sparse_input/valuesGatherNdKdnn/input_from_feature_columns/input_layer/nationality_indicator/ExpandDimsXdnn/input_from_feature_columns/input_layer/nationality_indicator/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:’’’’’’’’’
ē
\dnn/input_from_feature_columns/input_layer/nationality_indicator/to_sparse_input/dense_shapeShapeKdnn/input_from_feature_columns/input_layer/nationality_indicator/ExpandDims*
T0*
out_type0	*
_output_shapes
:

Ydnn/input_from_feature_columns/input_layer/nationality_indicator/nationality_lookup/ConstConst*
_output_shapes
:*{
valuerBpBotherBirishBbritishB	taiwaneseB
indonesianBjapaneseBchineseBsingaporeanBamericanBalgerianBthai*
dtype0

Xdnn/input_from_feature_columns/input_layer/nationality_indicator/nationality_lookup/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
”
_dnn/input_from_feature_columns/input_layer/nationality_indicator/nationality_lookup/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
”
_dnn/input_from_feature_columns/input_layer/nationality_indicator/nationality_lookup/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

Ydnn/input_from_feature_columns/input_layer/nationality_indicator/nationality_lookup/rangeRange_dnn/input_from_feature_columns/input_layer/nationality_indicator/nationality_lookup/range/startXdnn/input_from_feature_columns/input_layer/nationality_indicator/nationality_lookup/Size_dnn/input_from_feature_columns/input_layer/nationality_indicator/nationality_lookup/range/delta*
_output_shapes
:
ņ
[dnn/input_from_feature_columns/input_layer/nationality_indicator/nationality_lookup/ToInt64CastYdnn/input_from_feature_columns/input_layer/nationality_indicator/nationality_lookup/range*

DstT0	*
_output_shapes
:*

SrcT0
©
^dnn/input_from_feature_columns/input_layer/nationality_indicator/nationality_lookup/hash_tableHashTableV2*
value_dtype0	*
_output_shapes
: *
	key_dtype0
Æ
ddnn/input_from_feature_columns/input_layer/nationality_indicator/nationality_lookup/hash_table/ConstConst*
_output_shapes
: *
valueB	 R
’’’’’’’’’*
dtype0	
®
idnn/input_from_feature_columns/input_layer/nationality_indicator/nationality_lookup/hash_table/table_initInitializeTableV2^dnn/input_from_feature_columns/input_layer/nationality_indicator/nationality_lookup/hash_tableYdnn/input_from_feature_columns/input_layer/nationality_indicator/nationality_lookup/Const[dnn/input_from_feature_columns/input_layer/nationality_indicator/nationality_lookup/ToInt64*

Tkey0*

Tval0	
Ā
Rdnn/input_from_feature_columns/input_layer/nationality_indicator/hash_table_LookupLookupTableFindV2^dnn/input_from_feature_columns/input_layer/nationality_indicator/nationality_lookup/hash_tableWdnn/input_from_feature_columns/input_layer/nationality_indicator/to_sparse_input/valuesddnn/input_from_feature_columns/input_layer/nationality_indicator/nationality_lookup/hash_table/Const*	
Tin0*#
_output_shapes
:’’’’’’’’’*

Tout0	
§
\dnn/input_from_feature_columns/input_layer/nationality_indicator/SparseToDense/default_valueConst*
valueB	 R
’’’’’’’’’*
dtype0	*
_output_shapes
: 

Ndnn/input_from_feature_columns/input_layer/nationality_indicator/SparseToDenseSparseToDenseXdnn/input_from_feature_columns/input_layer/nationality_indicator/to_sparse_input/indices\dnn/input_from_feature_columns/input_layer/nationality_indicator/to_sparse_input/dense_shapeRdnn/input_from_feature_columns/input_layer/nationality_indicator/hash_table_Lookup\dnn/input_from_feature_columns/input_layer/nationality_indicator/SparseToDense/default_value*'
_output_shapes
:’’’’’’’’’*
Tindices0	*
T0	

Ndnn/input_from_feature_columns/input_layer/nationality_indicator/one_hot/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Pdnn/input_from_feature_columns/input_layer/nationality_indicator/one_hot/Const_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 

Ndnn/input_from_feature_columns/input_layer/nationality_indicator/one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 

Qdnn/input_from_feature_columns/input_layer/nationality_indicator/one_hot/on_valueConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Rdnn/input_from_feature_columns/input_layer/nationality_indicator/one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ļ
Hdnn/input_from_feature_columns/input_layer/nationality_indicator/one_hotOneHotNdnn/input_from_feature_columns/input_layer/nationality_indicator/SparseToDenseNdnn/input_from_feature_columns/input_layer/nationality_indicator/one_hot/depthQdnn/input_from_feature_columns/input_layer/nationality_indicator/one_hot/on_valueRdnn/input_from_feature_columns/input_layer/nationality_indicator/one_hot/off_value*
T0*+
_output_shapes
:’’’’’’’’’
©
Vdnn/input_from_feature_columns/input_layer/nationality_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
valueB:
ž’’’’’’’’*
dtype0

Ddnn/input_from_feature_columns/input_layer/nationality_indicator/SumSumHdnn/input_from_feature_columns/input_layer/nationality_indicator/one_hotVdnn/input_from_feature_columns/input_layer/nationality_indicator/Sum/reduction_indices*'
_output_shapes
:’’’’’’’’’*
T0
ŗ
Fdnn/input_from_feature_columns/input_layer/nationality_indicator/ShapeShapeDdnn/input_from_feature_columns/input_layer/nationality_indicator/Sum*
T0*
_output_shapes
:

Tdnn/input_from_feature_columns/input_layer/nationality_indicator/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
 
Vdnn/input_from_feature_columns/input_layer/nationality_indicator/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
 
Vdnn/input_from_feature_columns/input_layer/nationality_indicator/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ņ
Ndnn/input_from_feature_columns/input_layer/nationality_indicator/strided_sliceStridedSliceFdnn/input_from_feature_columns/input_layer/nationality_indicator/ShapeTdnn/input_from_feature_columns/input_layer/nationality_indicator/strided_slice/stackVdnn/input_from_feature_columns/input_layer/nationality_indicator/strided_slice/stack_1Vdnn/input_from_feature_columns/input_layer/nationality_indicator/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0

Pdnn/input_from_feature_columns/input_layer/nationality_indicator/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
¦
Ndnn/input_from_feature_columns/input_layer/nationality_indicator/Reshape/shapePackNdnn/input_from_feature_columns/input_layer/nationality_indicator/strided_slicePdnn/input_from_feature_columns/input_layer/nationality_indicator/Reshape/shape/1*
T0*
N*
_output_shapes
:

Hdnn/input_from_feature_columns/input_layer/nationality_indicator/ReshapeReshapeDdnn/input_from_feature_columns/input_layer/nationality_indicator/SumNdnn/input_from_feature_columns/input_layer/nationality_indicator/Reshape/shape*
T0*'
_output_shapes
:’’’’’’’’’

@dnn/input_from_feature_columns/input_layer/stress/ExpandDims/dimConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 
ė
<dnn/input_from_feature_columns/input_layer/stress/ExpandDims
ExpandDims+transform/transform/scale_by_min_max_15/add@dnn/input_from_feature_columns/input_layer/stress/ExpandDims/dim*
T0*'
_output_shapes
:’’’’’’’’’
£
7dnn/input_from_feature_columns/input_layer/stress/ShapeShape<dnn/input_from_feature_columns/input_layer/stress/ExpandDims*
_output_shapes
:*
T0

Ednn/input_from_feature_columns/input_layer/stress/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Gdnn/input_from_feature_columns/input_layer/stress/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0

Gdnn/input_from_feature_columns/input_layer/stress/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
§
?dnn/input_from_feature_columns/input_layer/stress/strided_sliceStridedSlice7dnn/input_from_feature_columns/input_layer/stress/ShapeEdnn/input_from_feature_columns/input_layer/stress/strided_slice/stackGdnn/input_from_feature_columns/input_layer/stress/strided_slice/stack_1Gdnn/input_from_feature_columns/input_layer/stress/strided_slice/stack_2*
_output_shapes
: *
Index0*
T0*
shrink_axis_mask

Adnn/input_from_feature_columns/input_layer/stress/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
ł
?dnn/input_from_feature_columns/input_layer/stress/Reshape/shapePack?dnn/input_from_feature_columns/input_layer/stress/strided_sliceAdnn/input_from_feature_columns/input_layer/stress/Reshape/shape/1*
_output_shapes
:*
T0*
N
õ
9dnn/input_from_feature_columns/input_layer/stress/ReshapeReshape<dnn/input_from_feature_columns/input_layer/stress/ExpandDims?dnn/input_from_feature_columns/input_layer/stress/Reshape/shape*
T0*'
_output_shapes
:’’’’’’’’’

Bdnn/input_from_feature_columns/input_layer/thorough/ExpandDims/dimConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 
ļ
>dnn/input_from_feature_columns/input_layer/thorough/ExpandDims
ExpandDims+transform/transform/scale_by_min_max_17/addBdnn/input_from_feature_columns/input_layer/thorough/ExpandDims/dim*
T0*'
_output_shapes
:’’’’’’’’’
§
9dnn/input_from_feature_columns/input_layer/thorough/ShapeShape>dnn/input_from_feature_columns/input_layer/thorough/ExpandDims*
T0*
_output_shapes
:

Gdnn/input_from_feature_columns/input_layer/thorough/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Idnn/input_from_feature_columns/input_layer/thorough/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Idnn/input_from_feature_columns/input_layer/thorough/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
±
Adnn/input_from_feature_columns/input_layer/thorough/strided_sliceStridedSlice9dnn/input_from_feature_columns/input_layer/thorough/ShapeGdnn/input_from_feature_columns/input_layer/thorough/strided_slice/stackIdnn/input_from_feature_columns/input_layer/thorough/strided_slice/stack_1Idnn/input_from_feature_columns/input_layer/thorough/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0

Cdnn/input_from_feature_columns/input_layer/thorough/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
’
Adnn/input_from_feature_columns/input_layer/thorough/Reshape/shapePackAdnn/input_from_feature_columns/input_layer/thorough/strided_sliceCdnn/input_from_feature_columns/input_layer/thorough/Reshape/shape/1*
_output_shapes
:*
T0*
N
ū
;dnn/input_from_feature_columns/input_layer/thorough/ReshapeReshape>dnn/input_from_feature_columns/input_layer/thorough/ExpandDimsAdnn/input_from_feature_columns/input_layer/thorough/Reshape/shape*'
_output_shapes
:’’’’’’’’’*
T0
x
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
¶
1dnn/input_from_feature_columns/input_layer/concatConcatV2Bdnn/input_from_feature_columns/input_layer/music_pref_none/ReshapeHdnn/input_from_feature_columns/input_layer/nationality_indicator/Reshape9dnn/input_from_feature_columns/input_layer/stress/Reshape;dnn/input_from_feature_columns/input_layer/thorough/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*
T0*
N*'
_output_shapes
:’’’’’’’’’
Å
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"      *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
:
·
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *o}R¾*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
·
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *o}R>*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0

Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0

>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
_output_shapes
: *
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
­
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*
_output_shapes
:	*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0

:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
_output_shapes
:	*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
„
dnn/hiddenlayer_0/kernel/part_0
VariableV2*
shape:	*
dtype0*
_output_shapes
:	*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
ė
&dnn/hiddenlayer_0/kernel/part_0/AssignAssigndnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:	
Æ
$dnn/hiddenlayer_0/kernel/part_0/readIdentitydnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:	*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
°
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0

dnn/hiddenlayer_0/bias/part_0
VariableV2*
dtype0*
_output_shapes	
:*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
shape:
Ö
$dnn/hiddenlayer_0/bias/part_0/AssignAssigndnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes	
:
„
"dnn/hiddenlayer_0/bias/part_0/readIdentitydnn/hiddenlayer_0/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes	
:
t
dnn/hiddenlayer_0/kernelIdentity$dnn/hiddenlayer_0/kernel/part_0/read*
T0*
_output_shapes
:	
¢
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*
T0*(
_output_shapes
:’’’’’’’’’
l
dnn/hiddenlayer_0/biasIdentity"dnn/hiddenlayer_0/bias/part_0/read*
_output_shapes	
:*
T0

dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
T0*(
_output_shapes
:’’’’’’’’’
l
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*(
_output_shapes
:’’’’’’’’’
[
dnn/zero_fraction/zeroConst*
_output_shapes
: *
valueB
 *    *
dtype0

dnn/zero_fraction/EqualEqualdnn/hiddenlayer_0/Reludnn/zero_fraction/zero*
T0*(
_output_shapes
:’’’’’’’’’
y
dnn/zero_fraction/CastCastdnn/zero_fraction/Equal*

DstT0*(
_output_shapes
:’’’’’’’’’*

SrcT0

h
dnn/zero_fraction/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
p
dnn/zero_fraction/MeanMeandnn/zero_fraction/Castdnn/zero_fraction/Const*
T0*
_output_shapes
: 
 
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values*
dtype0*
_output_shapes
: 
«
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/Mean*
T0*
_output_shapes
: 

$dnn/dnn/hiddenlayer_0/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_0/activation*
dtype0*
_output_shapes
: 

 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
_output_shapes
: 
Å
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"       *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
:
·
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *ųKF¾*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0
·
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ųKF>*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0

Hdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*
_output_shapes
:	 *
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0

>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
­
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	 

:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	 
„
dnn/hiddenlayer_1/kernel/part_0
VariableV2*
dtype0*
_output_shapes
:	 *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
shape:	 
ė
&dnn/hiddenlayer_1/kernel/part_0/AssignAssigndnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	 *
T0
Æ
$dnn/hiddenlayer_1/kernel/part_0/readIdentitydnn/hiddenlayer_1/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	 
®
/dnn/hiddenlayer_1/bias/part_0/Initializer/zerosConst*
_output_shapes
: *
valueB *    *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0

dnn/hiddenlayer_1/bias/part_0
VariableV2*
_output_shapes
: *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
shape: *
dtype0
Õ
$dnn/hiddenlayer_1/bias/part_0/AssignAssigndnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
: 
¤
"dnn/hiddenlayer_1/bias/part_0/readIdentitydnn/hiddenlayer_1/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
: 
t
dnn/hiddenlayer_1/kernelIdentity$dnn/hiddenlayer_1/kernel/part_0/read*
_output_shapes
:	 *
T0

dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Reludnn/hiddenlayer_1/kernel*'
_output_shapes
:’’’’’’’’’ *
T0
k
dnn/hiddenlayer_1/biasIdentity"dnn/hiddenlayer_1/bias/part_0/read*
T0*
_output_shapes
: 

dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*'
_output_shapes
:’’’’’’’’’ *
T0
k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:’’’’’’’’’ 
]
dnn/zero_fraction_1/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 

dnn/zero_fraction_1/EqualEqualdnn/hiddenlayer_1/Reludnn/zero_fraction_1/zero*'
_output_shapes
:’’’’’’’’’ *
T0
|
dnn/zero_fraction_1/CastCastdnn/zero_fraction_1/Equal*

SrcT0
*

DstT0*'
_output_shapes
:’’’’’’’’’ 
j
dnn/zero_fraction_1/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
v
dnn/zero_fraction_1/MeanMeandnn/zero_fraction_1/Castdnn/zero_fraction_1/Const*
T0*
_output_shapes
: 
 
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values*
dtype0*
_output_shapes
: 
­
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/Mean*
_output_shapes
: *
T0

$dnn/dnn/hiddenlayer_1/activation/tagConst*
_output_shapes
: *1
value(B& B dnn/dnn/hiddenlayer_1/activation*
dtype0

 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
_output_shapes
: 
Å
@dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"       *2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes
:
·
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *ģŃ¾*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes
: 
·
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ģŃ>*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0

Hdnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

: *
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0

>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/min*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes
: *
T0
¬
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/sub*
_output_shapes

: *
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0

:dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/min*
_output_shapes

: *
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0
£
dnn/hiddenlayer_2/kernel/part_0
VariableV2*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
shape
: *
dtype0*
_output_shapes

: 
ź
&dnn/hiddenlayer_2/kernel/part_0/AssignAssigndnn/hiddenlayer_2/kernel/part_0:dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

: 
®
$dnn/hiddenlayer_2/kernel/part_0/readIdentitydnn/hiddenlayer_2/kernel/part_0*
_output_shapes

: *
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0
®
/dnn/hiddenlayer_2/bias/part_0/Initializer/zerosConst*
valueB*    *0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
dtype0*
_output_shapes
:

dnn/hiddenlayer_2/bias/part_0
VariableV2*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
shape:*
dtype0*
_output_shapes
:
Õ
$dnn/hiddenlayer_2/bias/part_0/AssignAssigndnn/hiddenlayer_2/bias/part_0/dnn/hiddenlayer_2/bias/part_0/Initializer/zeros*
_output_shapes
:*
T0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0
¤
"dnn/hiddenlayer_2/bias/part_0/readIdentitydnn/hiddenlayer_2/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
_output_shapes
:
s
dnn/hiddenlayer_2/kernelIdentity$dnn/hiddenlayer_2/kernel/part_0/read*
_output_shapes

: *
T0

dnn/hiddenlayer_2/MatMulMatMuldnn/hiddenlayer_1/Reludnn/hiddenlayer_2/kernel*'
_output_shapes
:’’’’’’’’’*
T0
k
dnn/hiddenlayer_2/biasIdentity"dnn/hiddenlayer_2/bias/part_0/read*
T0*
_output_shapes
:

dnn/hiddenlayer_2/BiasAddBiasAdddnn/hiddenlayer_2/MatMuldnn/hiddenlayer_2/bias*'
_output_shapes
:’’’’’’’’’*
T0
k
dnn/hiddenlayer_2/ReluReludnn/hiddenlayer_2/BiasAdd*'
_output_shapes
:’’’’’’’’’*
T0
]
dnn/zero_fraction_2/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 

dnn/zero_fraction_2/EqualEqualdnn/hiddenlayer_2/Reludnn/zero_fraction_2/zero*
T0*'
_output_shapes
:’’’’’’’’’
|
dnn/zero_fraction_2/CastCastdnn/zero_fraction_2/Equal*

DstT0*'
_output_shapes
:’’’’’’’’’*

SrcT0

j
dnn/zero_fraction_2/ConstConst*
_output_shapes
:*
valueB"       *
dtype0
v
dnn/zero_fraction_2/MeanMeandnn/zero_fraction_2/Castdnn/zero_fraction_2/Const*
_output_shapes
: *
T0
 
2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_2/fraction_of_zero_values*
dtype0*
_output_shapes
: 
­
-dnn/dnn/hiddenlayer_2/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsdnn/zero_fraction_2/Mean*
T0*
_output_shapes
: 

$dnn/dnn/hiddenlayer_2/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_2/activation*
dtype0*
_output_shapes
: 

 dnn/dnn/hiddenlayer_2/activationHistogramSummary$dnn/dnn/hiddenlayer_2/activation/tagdnn/hiddenlayer_2/Relu*
_output_shapes
: 
·
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"      *+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
:
©
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *7æ*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0
©
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *7?*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
š
Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*
T0*+
_class!
loc:@dnn/logits/kernel/part_0
ž
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: 

7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:*
T0

3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
_output_shapes

:*
T0*+
_class!
loc:@dnn/logits/kernel/part_0

dnn/logits/kernel/part_0
VariableV2*
dtype0*
_output_shapes

:*+
_class!
loc:@dnn/logits/kernel/part_0*
shape
:
Ī
dnn/logits/kernel/part_0/AssignAssigndnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:

dnn/logits/kernel/part_0/readIdentitydnn/logits/kernel/part_0*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:
 
(dnn/logits/bias/part_0/Initializer/zerosConst*
valueB*    *)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
:

dnn/logits/bias/part_0
VariableV2*
shape:*
dtype0*
_output_shapes
:*)
_class
loc:@dnn/logits/bias/part_0
¹
dnn/logits/bias/part_0/AssignAssigndnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*
_output_shapes
:*
T0*)
_class
loc:@dnn/logits/bias/part_0

dnn/logits/bias/part_0/readIdentitydnn/logits/bias/part_0*
T0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:
e
dnn/logits/kernelIdentitydnn/logits/kernel/part_0/read*
_output_shapes

:*
T0
x
dnn/logits/MatMulMatMuldnn/hiddenlayer_2/Reludnn/logits/kernel*
T0*'
_output_shapes
:’’’’’’’’’
]
dnn/logits/biasIdentitydnn/logits/bias/part_0/read*
_output_shapes
:*
T0
s
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*
T0*'
_output_shapes
:’’’’’’’’’
]
dnn/zero_fraction_3/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 

dnn/zero_fraction_3/EqualEqualdnn/logits/BiasAdddnn/zero_fraction_3/zero*
T0*'
_output_shapes
:’’’’’’’’’
|
dnn/zero_fraction_3/CastCastdnn/zero_fraction_3/Equal*

SrcT0
*

DstT0*'
_output_shapes
:’’’’’’’’’
j
dnn/zero_fraction_3/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
v
dnn/zero_fraction_3/MeanMeandnn/zero_fraction_3/Castdnn/zero_fraction_3/Const*
_output_shapes
: *
T0

+dnn/dnn/logits/fraction_of_zero_values/tagsConst*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values*
dtype0*
_output_shapes
: 

&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_3/Mean*
_output_shapes
: *
T0
w
dnn/dnn/logits/activation/tagConst**
value!B Bdnn/dnn/logits/activation*
dtype0*
_output_shapes
: 
x
dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
_output_shapes
: 
c
!dnn/head/predictions/logits/ShapeShapednn/logits/BiasAdd*
_output_shapes
:*
T0
w
5dnn/head/predictions/logits/assert_rank_at_least/rankConst*
_output_shapes
: *
value	B :*
dtype0
g
_dnn/head/predictions/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
X
Pdnn/head/predictions/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
n
dnn/head/predictions/logisticSigmoiddnn/logits/BiasAdd*
T0*'
_output_shapes
:’’’’’’’’’
r
dnn/head/predictions/zeros_like	ZerosLikednn/logits/BiasAdd*'
_output_shapes
:’’’’’’’’’*
T0
u
*dnn/head/predictions/two_class_logits/axisConst*
dtype0*
_output_shapes
: *
valueB :
’’’’’’’’’
Ķ
%dnn/head/predictions/two_class_logitsConcatV2dnn/head/predictions/zeros_likednn/logits/BiasAdd*dnn/head/predictions/two_class_logits/axis*
T0*
N*'
_output_shapes
:’’’’’’’’’

"dnn/head/predictions/probabilitiesSoftmax%dnn/head/predictions/two_class_logits*
T0*'
_output_shapes
:’’’’’’’’’
s
(dnn/head/predictions/class_ids/dimensionConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 
§
dnn/head/predictions/class_idsArgMax%dnn/head/predictions/two_class_logits(dnn/head/predictions/class_ids/dimension*
T0*#
_output_shapes
:’’’’’’’’’
n
#dnn/head/predictions/ExpandDims/dimConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 
¤
dnn/head/predictions/ExpandDims
ExpandDimsdnn/head/predictions/class_ids#dnn/head/predictions/ExpandDims/dim*'
_output_shapes
:’’’’’’’’’*
T0	

 dnn/head/predictions/str_classesAsStringdnn/head/predictions/ExpandDims*
T0	*'
_output_shapes
:’’’’’’’’’
`
dnn/head/ShapeShape"dnn/head/predictions/probabilities*
_output_shapes
:*
T0
f
dnn/head/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
h
dnn/head/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
h
dnn/head/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ś
dnn/head/strided_sliceStridedSlicednn/head/Shapednn/head/strided_slice/stackdnn/head/strided_slice/stack_1dnn/head/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0
V
dnn/head/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
V
dnn/head/range/limitConst*
dtype0*
_output_shapes
: *
value	B :
V
dnn/head/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
u
dnn/head/rangeRangednn/head/range/startdnn/head/range/limitdnn/head/range/delta*
_output_shapes
:
R
dnn/head/AsStringAsStringdnn/head/range*
T0*
_output_shapes
:
Y
dnn/head/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
v
dnn/head/ExpandDims
ExpandDimsdnn/head/AsStringdnn/head/ExpandDims/dim*
T0*
_output_shapes

:
[
dnn/head/Tile/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 

dnn/head/Tile/multiplesPackdnn/head/strided_slicednn/head/Tile/multiples/1*
T0*
N*
_output_shapes
:
u
dnn/head/TileTilednn/head/ExpandDimsdnn/head/Tile/multiples*
T0*'
_output_shapes
:’’’’’’’’’
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_1/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_67d0564eb7444e6eb58ac63c89f40925/part*
dtype0
j
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
value	B : *
dtype0

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
ŗ
save_1/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*Ü
valueŅBĻ	Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0
Ś
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*y
valuepBn	B	128 0,128B14 128 0,14:0,128B32 0,32B128 32 0,128:0,32B4 0,4B32 4 0,32:0,4B1 0,1B4 1 0,4:0,1B *
dtype0*
_output_shapes
:	
ŗ
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slices"dnn/hiddenlayer_0/bias/part_0/read$dnn/hiddenlayer_0/kernel/part_0/read"dnn/hiddenlayer_1/bias/part_0/read$dnn/hiddenlayer_1/kernel/part_0/read"dnn/hiddenlayer_2/bias/part_0/read$dnn/hiddenlayer_2/kernel/part_0/readdnn/logits/bias/part_0/readdnn/logits/kernel/part_0/readglobal_step"/device:CPU:0*
dtypes
2		
Ø
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
¦
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
T0*
N*
_output_shapes
:
{
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0

save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
½
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*Ü
valueŅBĻ	Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0
Ż
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*y
valuepBn	B	128 0,128B14 128 0,14:0,128B32 0,32B128 32 0,128:0,32B4 0,4B32 4 0,32:0,4B1 0,1B4 1 0,4:0,1B *
dtype0*
_output_shapes
:	
ņ
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*[
_output_shapesI
G::	: :	 :: :::*
dtypes
2		
 
save_1/AssignAssigndnn/hiddenlayer_0/bias/part_0save_1/RestoreV2*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes	
:
¬
save_1/Assign_1Assigndnn/hiddenlayer_0/kernel/part_0save_1/RestoreV2:1*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:	
£
save_1/Assign_2Assigndnn/hiddenlayer_1/bias/part_0save_1/RestoreV2:2*
_output_shapes
: *
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0
¬
save_1/Assign_3Assigndnn/hiddenlayer_1/kernel/part_0save_1/RestoreV2:3*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	 
£
save_1/Assign_4Assigndnn/hiddenlayer_2/bias/part_0save_1/RestoreV2:4*
T0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
_output_shapes
:
«
save_1/Assign_5Assigndnn/hiddenlayer_2/kernel/part_0save_1/RestoreV2:5*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

: 

save_1/Assign_6Assigndnn/logits/bias/part_0save_1/RestoreV2:6*
T0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:

save_1/Assign_7Assigndnn/logits/kernel/part_0save_1/RestoreV2:7*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:*
T0
{
save_1/Assign_8Assignglobal_stepsave_1/RestoreV2:8*
_output_shapes
: *
T0	*
_class
loc:@global_step
¼
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8
1
save_1/restore_allNoOp^save_1/restore_shard

initNoOp

init_all_tablesNoOpj^dnn/input_from_feature_columns/input_layer/nationality_indicator/nationality_lookup/hash_table/table_init

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
R
save_2/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel

save_2/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_d2fcaf21d9cf40878cb42b2f68009928/part
j
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
_output_shapes
: *
N
S
save_2/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
m
save_2/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards"/device:CPU:0*
_output_shapes
: 
ŗ
save_2/SaveV2/tensor_namesConst"/device:CPU:0*Ü
valueŅBĻ	Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:	
Ś
save_2/SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:	*y
valuepBn	B	128 0,128B14 128 0,14:0,128B32 0,32B128 32 0,128:0,32B4 0,4B32 4 0,32:0,4B1 0,1B4 1 0,4:0,1B 
ŗ
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slices"dnn/hiddenlayer_0/bias/part_0/read$dnn/hiddenlayer_0/kernel/part_0/read"dnn/hiddenlayer_1/bias/part_0/read$dnn/hiddenlayer_1/kernel/part_0/read"dnn/hiddenlayer_2/bias/part_0/read$dnn/hiddenlayer_2/kernel/part_0/readdnn/logits/bias/part_0/readdnn/logits/kernel/part_0/readglobal_step"/device:CPU:0*
dtypes
2		
Ø
save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2"/device:CPU:0*)
_class
loc:@save_2/ShardedFilename*
_output_shapes
: *
T0
¦
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency"/device:CPU:0*
T0*
N*
_output_shapes
:
{
save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const"/device:CPU:0

save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
½
save_2/RestoreV2/tensor_namesConst"/device:CPU:0*Ü
valueŅBĻ	Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:	
Ż
!save_2/RestoreV2/shape_and_slicesConst"/device:CPU:0*y
valuepBn	B	128 0,128B14 128 0,14:0,128B32 0,32B128 32 0,128:0,32B4 0,4B32 4 0,32:0,4B1 0,1B4 1 0,4:0,1B *
dtype0*
_output_shapes
:	
ņ
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices"/device:CPU:0*[
_output_shapesI
G::	: :	 :: :::*
dtypes
2		
 
save_2/AssignAssigndnn/hiddenlayer_0/bias/part_0save_2/RestoreV2*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes	
:
¬
save_2/Assign_1Assigndnn/hiddenlayer_0/kernel/part_0save_2/RestoreV2:1*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:	
£
save_2/Assign_2Assigndnn/hiddenlayer_1/bias/part_0save_2/RestoreV2:2*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
: *
T0
¬
save_2/Assign_3Assigndnn/hiddenlayer_1/kernel/part_0save_2/RestoreV2:3*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	 
£
save_2/Assign_4Assigndnn/hiddenlayer_2/bias/part_0save_2/RestoreV2:4*
_output_shapes
:*
T0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0
«
save_2/Assign_5Assigndnn/hiddenlayer_2/kernel/part_0save_2/RestoreV2:5*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

: 

save_2/Assign_6Assigndnn/logits/bias/part_0save_2/RestoreV2:6*
T0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:

save_2/Assign_7Assigndnn/logits/kernel/part_0save_2/RestoreV2:7*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:*
T0
{
save_2/Assign_8Assignglobal_stepsave_2/RestoreV2:8*
T0	*
_class
loc:@global_step*
_output_shapes
: 
¼
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_2^save_2/Assign_3^save_2/Assign_4^save_2/Assign_5^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8
1
save_2/restore_allNoOp^save_2/restore_shard"B
save_2/Const:0save_2/Identity:0save_2/restore_all (5 @F8"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"×
	summariesÉ
Ę
/dnn/dnn/hiddenlayer_0/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_0/activation:0
/dnn/dnn/hiddenlayer_1/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_1/activation:0
/dnn/dnn/hiddenlayer_2/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_2/activation:0
(dnn/dnn/logits/fraction_of_zero_values:0
dnn/dnn/logits/activation:0"ć
trainable_variablesĖČ
Ū
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign&dnn/hiddenlayer_0/kernel/part_0/read:0"(
dnn/hiddenlayer_0/kernel  "2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:0
Å
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign$dnn/hiddenlayer_0/bias/part_0/read:0"#
dnn/hiddenlayer_0/bias "21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:0
Ū
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign&dnn/hiddenlayer_1/kernel/part_0/read:0"(
dnn/hiddenlayer_1/kernel   " 2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:0
Ć
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign$dnn/hiddenlayer_1/bias/part_0/read:0"!
dnn/hiddenlayer_1/bias  " 21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:0
Ł
!dnn/hiddenlayer_2/kernel/part_0:0&dnn/hiddenlayer_2/kernel/part_0/Assign&dnn/hiddenlayer_2/kernel/part_0/read:0"&
dnn/hiddenlayer_2/kernel   " 2<dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform:0
Ć
dnn/hiddenlayer_2/bias/part_0:0$dnn/hiddenlayer_2/bias/part_0/Assign$dnn/hiddenlayer_2/bias/part_0/read:0"!
dnn/hiddenlayer_2/bias "21dnn/hiddenlayer_2/bias/part_0/Initializer/zeros:0
¶
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assigndnn/logits/kernel/part_0/read:0"
dnn/logits/kernel  "25dnn/logits/kernel/part_0/Initializer/random_uniform:0
 
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assigndnn/logits/bias/part_0/read:0"
dnn/logits/bias "2*dnn/logits/bias/part_0/Initializer/zeros:0"³
	variables„¢
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
Ū
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign&dnn/hiddenlayer_0/kernel/part_0/read:0"(
dnn/hiddenlayer_0/kernel  "2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:0
Å
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign$dnn/hiddenlayer_0/bias/part_0/read:0"#
dnn/hiddenlayer_0/bias "21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:0
Ū
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign&dnn/hiddenlayer_1/kernel/part_0/read:0"(
dnn/hiddenlayer_1/kernel   " 2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:0
Ć
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign$dnn/hiddenlayer_1/bias/part_0/read:0"!
dnn/hiddenlayer_1/bias  " 21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:0
Ł
!dnn/hiddenlayer_2/kernel/part_0:0&dnn/hiddenlayer_2/kernel/part_0/Assign&dnn/hiddenlayer_2/kernel/part_0/read:0"&
dnn/hiddenlayer_2/kernel   " 2<dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform:0
Ć
dnn/hiddenlayer_2/bias/part_0:0$dnn/hiddenlayer_2/bias/part_0/Assign$dnn/hiddenlayer_2/bias/part_0/read:0"!
dnn/hiddenlayer_2/bias "21dnn/hiddenlayer_2/bias/part_0/Initializer/zeros:0
¶
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assigndnn/logits/kernel/part_0/read:0"
dnn/logits/kernel  "25dnn/logits/kernel/part_0/Initializer/random_uniform:0
 
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assigndnn/logits/bias/part_0/read:0"
dnn/logits/bias "2*dnn/logits/bias/part_0/Initializer/zeros:0" 
legacy_init_op


group_deps"
table_initializerm
k
idnn/input_from_feature_columns/input_layer/nationality_indicator/nationality_lookup/hash_table/table_init*×
predictĖ
.
activity"
Placeholder_1:0’’’’’’’’’
'
age 
Placeholder:0’’’’’’’’’B
logistic6
dnn/head/predictions/logistic:0’’’’’’’’’E
	class_ids8
!dnn/head/predictions/ExpandDims:0	’’’’’’’’’L
probabilities;
$dnn/head/predictions/probabilities:0’’’’’’’’’D
classes9
"dnn/head/predictions/str_classes:0’’’’’’’’’5
logits+
dnn/logits/BiasAdd:0’’’’’’’’’tensorflow/serving/predict