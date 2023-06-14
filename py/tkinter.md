- window : container in which all GUI elements(widgets) live. `window = tkinter.Tk()`
- widgets
- geometry manager
- event handlers

`window.mainloop()` - infinite loop, wait for an event and process the event as long as the window is not closed.

geometry managers - pack, grid, place    

widgets - button, entry, frame (widget container), ..   
`widget_handler = tkinter.Widget(master=parent_window, option = value)`
`widget_handler.something()`  

This only creates the widgets. Has to be added to the window by one of the geometry managers:   
`widget.pack( side=  )`
`widget.grid( row= , column=)`
`widget.place( x=, y=)` 


`def event_handler(event)` 
`widget_handler.bind("<event_name>, event_handler")`



<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEyMzc5NTcxMzZdfQ==
-->