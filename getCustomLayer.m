function layer = getCustomLayer()
layer.name = 'loss' ;
layer.type = 'custom' ;
layer.forward = @forward ;
layer.backward = @backward ;
layer.class = [] ;

function res_ =  forward(layer, res, res_)
res_.x = customLayerForward(res.x, layer.class) ;

function res = backward(layer, res, res_)
res.dzdx = customLayerBackward(res.x, layer.class, res_.dzdx) ;


