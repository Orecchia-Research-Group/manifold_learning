try:
    assert not np.all(ico.check_if_point_in_face(point,origin_face))
except:

    """fig = plt.figure()
    ax = Axes3D(fig)
    for idx,face_node in enumerate(list(face_graph.nodes())):
        plot_color = 'grey'    
        u,v,w = face_dict[face_node].return_coors()
        for x,y in [[u,v],[v,w],[u,w]]:
            ax.plot([x[0],y[0]],[x[1],y[1]],[x[2],y[2]],color=plot_color)
        for v in face_dict[face_node].return_coors():
            ax.scatter(v[0],v[1],v[2],color=plot_color)
    # plot origin face 
    plot_color = 'red'    
    u,v,w = origin_face.return_coors()
    for x,y in [[u,v],[v,w],[u,w]]:
        ax.plot([x[0],y[0]],[x[1],y[1]],[x[2],y[2]],color=plot_color)
    for v in origin_face.return_coors():
        ax.scatter(v[0],v[1],v[2],color=plot_color)"""
                                    

    # plot Transformed origin face and point
    plot_color='orange'
    transformation = origin_face.transform_face_by_chart()
    """u,v,w = transformation.return_coors()
    for x,y in [[u,v],[v,w],[u,w]]:
        ax.plot([x[0],y[0]],[x[1],y[1]],[x[2],y[2]],color=plot_color)
    for v in transformation.return_coors():
        ax.scatter(v[0],v[1],v[2],color=plot_color)"""

    pt_in_chart = ico.euclidean2chart(point,origin_face)
    if ico.righthand_face(origin_face):
        transformed_point = np.array([pt_in_chart[0],pt_in_chart[1],1.0])
    else:
        transformed_point = np.array([pt_in_chart[0],pt_in_chart[1],-1.0])

    """ax.quiver(0,0,0,point[0],point[1],point[2],color='green')
    ax.quiver(0,0,0,transformed_point[0],transformed_point[1],transformed_point[2],color='blue')


    ax.axes.set_xlim3d(left=-2, right=2) 
    ax.axes.set_ylim3d(bottom=-2, top=2) 
    ax.axes.set_zlim3d(bottom=-2, top=2) 
    plt.show()"""
    
    fig = plt.figure()
    ## Plot in chart coordinates
    plt.plot([pt_in_chart[0]],[pt_in_chart[1]],'o',label='point')
    u,v,w = transformation.return_coors()
    for x,y in [[u,v],[v,w],[u,w]]:
        plt.plot([x[0],y[0]],[x[1],y[1]])
    plt.legend()
    plt.show()

    print('Is ',point,' in this chart? ',ico.check_if_point_in_chart(point,origin_face))
    print('Is ',point,' in this face? ',np.all(ico.check_if_point_in_face(point,origin_face)))

    return fig

        assert np.all(ico.check_if_point_in_face(point,destination_face))
        

